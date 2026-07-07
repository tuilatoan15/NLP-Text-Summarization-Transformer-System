#!/usr/bin/env python3
"""Reusable UML sequence-diagram builder for draw.io (activation continuity)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MessageKind = Literal["sync", "async", "return", "self"]

BW_VERTEX = "fillColor=#ffffff;strokeColor=#000000;fontColor=#000000"
BW_TEXT = "strokeColor=none;fillColor=none;fontColor=#000000"
BW_EDGE = "strokeColor=#000000;fontColor=#000000"
BW_LIFELINE = "endArrow=none;dashed=1;html=1;strokeColor=#000000"
BW_ACT = "fillColor=#ffffff;strokeColor=#000000;opacity=60;rounded=0;whiteSpace=wrap;html=1"

ACT_W = 10
ACT_NEST = 3
MIN_ACT_H = 10
RETURN_OFFSET = 8
SELF_LOOP_H = 18
SELF_LOOP_W = 50


@dataclass
class Participant:
    pid: str
    label: str
    x: int
    kind: str = "box"  # box | actor


@dataclass
class Message:
    src: str
    dst: str
    label: str
    kind: MessageKind = "sync"


@dataclass
class _ActPeriod:
    y0: int
    y1: int
    depth: int = 0


@dataclass
class SequenceDiagram:
    title: str
    participants: list[Participant]
    messages: list[Message]
    y_start: int = 120
    y_gap: int = 30
    page_w: int = 1680
    extra_frames: list[dict] = field(default_factory=list)

    def _x(self) -> dict[str, int]:
        return {p.pid: p.x for p in self.participants}

    def _assign_y(self) -> list[int]:
        x = self._x()
        last: dict[str, int] = {p.pid: self.y_start - self.y_gap for p in self.participants}
        ys: list[int] = []
        cursor = self.y_start
        for msg in self.messages:
            if msg.kind == "self":
                y = max(cursor, last[msg.src] + 12)
            elif msg.kind == "return":
                y = max(cursor, last[msg.src] + RETURN_OFFSET, last[msg.dst] + RETURN_OFFSET)
            elif msg.kind == "async":
                y = max(cursor, last[msg.src] + 4)
            else:
                y = cursor
            y = max(y, last[msg.src] + (4 if msg.src == msg.dst else 0))
            ys.append(y)
            last[msg.src] = y
            if msg.dst != msg.src:
                last[msg.dst] = y
            cursor = y + self.y_gap
        return ys

    def _compute_activations(self, ys: list[int]) -> dict[str, list[_ActPeriod]]:
        """Build activation periods with nested depth on re-entry while still active."""
        stacks: dict[str, list[_ActPeriod]] = {p.pid: [] for p in self.participants}
        closed: dict[str, list[_ActPeriod]] = {p.pid: [] for p in self.participants}

        def touch(pid: str, y: int, depth: int) -> None:
            stack = stacks[pid]
            if stack and stack[-1].depth == depth:
                stack[-1].y1 = max(stack[-1].y1, y + MIN_ACT_H)
            else:
                if stack:
                    closed[pid].append(stack.pop())
                stack.append(_ActPeriod(y0=y, y1=y + MIN_ACT_H, depth=depth))

        def extend(pid: str, y: int) -> None:
            if stacks[pid]:
                stacks[pid][-1].y1 = max(stacks[pid][-1].y1, y + MIN_ACT_H)
            else:
                touch(pid, y, 0)

        def close_top(pid: str, y: int) -> None:
            if stacks[pid]:
                stacks[pid][-1].y1 = max(stacks[pid][-1].y1, y + MIN_ACT_H)
                closed[pid].append(stacks[pid].pop())

        for i, (msg, y) in enumerate(zip(self.messages, ys)):
            if msg.kind == "self":
                extend(msg.src, y + SELF_LOOP_H)
                continue

            extend(msg.src, y)
            if msg.kind == "sync":
                callee_depth = len(stacks[msg.dst])
                touch(msg.dst, y, callee_depth)
                nxt = self.messages[i + 1] if i + 1 < len(self.messages) else None
                if not (
                    nxt
                    and nxt.src == msg.dst
                    and nxt.dst == msg.src
                    and nxt.kind == "return"
                ):
                    close_top(msg.dst, y)
            elif msg.kind in ("return", "async"):
                extend(msg.dst, y)
            if msg.kind == "return":
                close_top(msg.src, y)
            elif msg.kind == "async":
                close_top(msg.src, y)

        for pid in stacks:
            while stacks[pid]:
                closed[pid].append(stacks[pid].pop())

        # Merge adjacent periods at same depth (visual continuity)
        merged: dict[str, list[_ActPeriod]] = {}
        for pid, periods in closed.items():
            periods.sort(key=lambda p: (p.y0, p.depth))
            out: list[_ActPeriod] = []
            for p in periods:
                if out and p.depth == out[-1].depth and p.y0 <= out[-1].y1 + self.y_gap:
                    out[-1].y1 = max(out[-1].y1, p.y1)
                else:
                    out.append(_ActPeriod(p.y0, p.y1, p.depth))
            merged[pid] = out
        return merged

    def _find_period(self, periods: list[_ActPeriod], y: int) -> _ActPeriod | None:
        for p in periods:
            if p.y0 <= y <= p.y1 + 2:
                return p
        best: _ActPeriod | None = None
        for p in periods:
            if p.y0 <= y and (best is None or p.y0 > best.y0):
                best = p
        return best

    def _rel_y(self, period: _ActPeriod, y: int) -> float:
        h = max(period.y1 - period.y0, MIN_ACT_H)
        rel = (y - period.y0) / h
        return max(0.02, min(0.98, rel))

    def _act_x(self, pid: str, depth: int, x_map: dict[str, int]) -> int:
        return x_map[pid] - ACT_W // 2 + depth * ACT_NEST

    def build_drawio(self, diagram_id: str, diagram_name: str) -> str:
        x_map = self._x()
        ys = self._assign_y()
        act_periods = self._compute_activations(ys)
        bottom_y = max(ys) + 80 if ys else self.y_start + 80

        cells: list[str] = []
        cid = 0
        act_ids: dict[tuple[str, int, int], str] = {}

        def nid(prefix: str = "c") -> str:
            nonlocal cid
            cid += 1
            return f"{prefix}{cid}"

        def add(cell: str) -> str:
            cells.append(cell)
            return cell

        add('<mxCell id="0" />')
        add('<mxCell id="1" parent="0" />')

        add(
            f'<mxCell id="{nid("t")}" value="{self.title}" '
            f'style="{BW_TEXT};text;html=1;align=center;verticalAlign=middle;'
            f'fontSize=14;fontStyle=1" vertex="1" parent="1">'
            f'<mxGeometry x="{self.page_w // 2 - 450}" y="10" width="900" height="30" as="geometry" />'
            f"</mxCell>"
        )

        box_w = 130
        header_y = 60
        for p in self.participants:
            if p.kind == "actor":
                add(
                    f'<mxCell id="{nid(p.pid)}" value="{p.label}" '
                    f'style="{BW_VERTEX};shape=umlActor;verticalLabelPosition=bottom;'
                    f'verticalAlign=top;html=1;outlineConnect=0" vertex="1" parent="1">'
                    f'<mxGeometry x="{p.x - 20}" y="{header_y - 10}" width="40" height="60" as="geometry" />'
                    f"</mxCell>"
                )
            else:
                add(
                    f'<mxCell id="{nid(p.pid)}" value="{p.label}" '
                    f'style="{BW_VERTEX};rounded=0;whiteSpace=wrap;html=1;fontStyle=1" '
                    f'vertex="1" parent="1">'
                    f'<mxGeometry x="{p.x - box_w // 2}" y="{header_y}" width="{box_w}" '
                    f'height="40" as="geometry" />'
                    f"</mxCell>"
                )

        ll_ids: dict[str, str] = {}
        for p in self.participants:
            ll_id = nid(f"ll-{p.pid}")
            ll_ids[p.pid] = ll_id
            add(
                f'<mxCell id="{ll_id}" style="{BW_LIFELINE}" edge="1" parent="1">'
                f'<mxGeometry relative="1" as="geometry">'
                f'<mxPoint x="{p.x}" y="{self.y_start}" as="sourcePoint" />'
                f'<mxPoint x="{p.x}" y="{bottom_y}" as="targetPoint" />'
                f"</mxGeometry></mxCell>"
            )

        for p in self.participants:
            for idx, period in enumerate(act_periods.get(p.pid, [])):
                ax = self._act_x(p.pid, period.depth, x_map)
                h = max(period.y1 - period.y0, MIN_ACT_H)
                aid = nid(f"act-{p.pid}")
                act_ids[(p.pid, period.y0, period.depth)] = aid
                add(
                    f'<mxCell id="{aid}" value="" style="{BW_ACT}" vertex="1" parent="1">'
                    f'<mxGeometry x="{ax}" y="{period.y0}" width="{ACT_W}" height="{h}" as="geometry" />'
                    f"</mxCell>"
                )

        def resolve_act(pid: str, y: int) -> tuple[str, _ActPeriod]:
            periods = act_periods.get(pid, [])
            period = self._find_period(periods, y)
            if period is None:
                period = _ActPeriod(y0=y, y1=y + MIN_ACT_H, depth=0)
                ax = self._act_x(pid, 0, x_map)
                aid = nid(f"act-{pid}")
                add(
                    f'<mxCell id="{aid}" value="" style="{BW_ACT}" vertex="1" parent="1">'
                    f'<mxGeometry x="{ax}" y="{period.y0}" width="{ACT_W}" height="{MIN_ACT_H}" as="geometry" />'
                    f"</mxCell>"
                )
                return aid, period
            key = (pid, period.y0, period.depth)
            aid = act_ids.get(key)
            if aid is None:
                for k, v in act_ids.items():
                    if k[0] == pid and k[2] == period.depth:
                        aid = v
                        break
            if aid is None:
                ax = self._act_x(pid, period.depth, x_map)
                aid = nid(f"act-{pid}")
                h = max(period.y1 - period.y0, MIN_ACT_H)
                add(
                    f'<mxCell id="{aid}" value="" style="{BW_ACT}" vertex="1" parent="1">'
                    f'<mxGeometry x="{ax}" y="{period.y0}" width="{ACT_W}" height="{h}" as="geometry" />'
                    f"</mxCell>"
                )
            return aid, period

        def edge_style(kind: MessageKind) -> str:
            base = (
                f"{BW_EDGE};html=1;verticalAlign=bottom;align=center;"
                "labelBackgroundColor=none;spacingTop=-6;rounded=0;fontSize=11;"
            )
            if kind in ("async", "return"):
                return base + "endArrow=open;dashed=1;"
            if kind == "self":
                return base + "endArrow=block;edgeStyle=orthogonalEdgeStyle;"
            return base + "endArrow=block;"

        for i, (msg, y) in enumerate(zip(self.messages, ys)):
            num = i + 1
            text = f"{num}. {msg.label}"
            style = edge_style(msg.kind)
            sx, dx = x_map[msg.src], x_map[msg.dst]

            if msg.kind == "self":
                src_act, period = resolve_act(msg.src, y)
                ry = self._rel_y(period, y)
                style_self = (
                    style + f"exitX=1;exitY={ry:.4f};exitDx=0;exitDy=0;"
                    f"entryX=1;entryY={ry:.4f};entryDx=0;entryDy=0;"
                )
                add(
                    f'<mxCell id="{nid("m")}" value="{text}" style="{style_self}" edge="1" parent="1" '
                    f'source="{src_act}" target="{src_act}">'
                    f'<mxGeometry x="0.15" relative="1" as="geometry">'
                    f'<mxPoint x="{sx + ACT_W}" y="{y}" as="sourcePoint" />'
                    f'<mxPoint x="{sx + ACT_W + SELF_LOOP_W}" y="{y + SELF_LOOP_H}" as="targetPoint" />'
                    f"</mxGeometry></mxCell>"
                )
                continue

            src_act, src_p = resolve_act(msg.src, y)
            dst_act, dst_p = resolve_act(msg.dst, y)
            src_ry = self._rel_y(src_p, y)
            dst_ry = self._rel_y(dst_p, y)

            if sx < dx:
                src_ex, dst_ex = "exitX=1", "entryX=0"
            else:
                src_ex, dst_ex = "exitX=0", "entryX=1"

            ortho = ""
            if abs(sx - dx) > 400 and msg.kind == "sync":
                mid_y = y + 14
                ortho = (
                    f"<Array as=\"points\">"
                    f'<mxPoint x="{sx}" y="{mid_y}" />'
                    f'<mxPoint x="{dx}" y="{mid_y}" />'
                    f"</Array>"
                )
                style += "edgeStyle=orthogonalEdgeStyle;"

            style_edge = (
                style + f"{src_ex};exitY={src_ry:.4f};exitDx=0;exitDy=0;"
                f"{dst_ex};entryY={dst_ry:.4f};entryDx=0;entryDy=0;"
            )
            add(
                f'<mxCell id="{nid("m")}" value="{text}" style="{style_edge}" edge="1" parent="1" '
                f'source="{src_act}" target="{dst_act}">'
                f'<mxGeometry relative="1" as="geometry">'
                f'<mxPoint x="{sx}" y="{y}" as="sourcePoint" />'
                f'<mxPoint x="{dx}" y="{y}" as="targetPoint" />'
                f"{ortho}"
                f"</mxGeometry></mxCell>"
            )

        for frame in self.extra_frames:
            add(
                f'<mxCell id="{nid("fr")}" value="{frame["label"]}" '
                f'style="{BW_VERTEX};shape=umlFrame;whiteSpace=wrap;html=1;'
                f'width=160;height=30;boundedLbl=1;backgroundOutline=1;size=20;fontSize=11" '
                f'vertex="1" parent="1">'
                f'<mxGeometry x="{frame["x"]}" y="{frame["y"]}" width="{frame["w"]}" '
                f'height="{frame["h"]}" as="geometry" />'
                f"</mxCell>"
            )

        page_h = bottom_y + 30
        inner = "\n        ".join(cells)
        return f"""<mxfile host="app.diagrams.net" modified="2026-07-04T04:00:00.000Z" agent="UML-Seq-Builder" version="22.1.0" type="device">
  <diagram id="{diagram_id}" name="{diagram_name}">
    <mxGraphModel dx="{self.page_w}" dy="{page_h}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.page_w}" pageHeight="{page_h}" math="0" shadow="0">
      <root>
        {inner}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
