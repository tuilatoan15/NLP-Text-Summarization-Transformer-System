# 🎨 Model Settings - Tài liệu Phát triển

## 📋 Tổng quan

Hệ thống **Model Settings** mới được thiết kế với các nguyên tắc:
- ✅ **Dễ sử dụng** - Giao diện trực quan cho người dùng phổ thông
- ✅ **Chuyên nghiệp** - Thiết kế hiện đại như HuggingFace/OpenAI
- ✅ **Hiệu năng** - Giảm số lượng tham số (chỉ giữ lại cần thiết)
- ✅ **Linh hoạt** - Dễ mở rộng và tích hợp

## 🏗️ Cấu trúc Tệp

```
frontend/src/
├── pages/
│   └── ModelSettings.jsx          # Component chính (UI/UX)
├── types/
│   └── modelSettings.ts           # TypeScript interfaces
├── hooks/
│   └── useModelSettings.ts        # Custom hook
├── context/
│   └── ModelSettingsContext.tsx   # Context API provider
└── utils/
    └── settingsUtils.ts           # Utility functions
```

## 🔧 Cách Sử Dụng

### 1. Trong Component (Local State)

```jsx
import ModelSettings from './pages/ModelSettings';

function App() {
  return <ModelSettings />;
}
```

### 2. Sử dụng Hook (Recommended cho non-local state)

```jsx
import useModelSettings from './hooks/useModelSettings';

function MyComponent() {
  const {
    settings,
    updateSetting,
    saveSettings,
    isDirty,
  } = useModelSettings();

  return (
    <div>
      <p>Temperature: {settings.temperature}</p>
      <button onClick={() => updateSetting('temperature', 0.8)}>
        Update
      </button>
    </div>
  );
}
```

### 3. Sử dụng Context (App-wide)

```jsx
import { ModelSettingsProvider, useModelSettingsContext } from './context/ModelSettingsContext';

function App() {
  return (
    <ModelSettingsProvider>
      <YourApp />
    </ModelSettingsProvider>
  );
}

function SomeComponent() {
  const { settings } = useModelSettingsContext();
  return <p>Current model: {settings.abstractiveModel}</p>;
}
```

## 📊 Các Tham Số Được Hỗ Trợ

### Abstractive (Tóm tắt)

| Tham Số | Phạm vi | Mặc định | Mô tả |
|---------|--------|---------|-------|
| `temperature` | 0 - 2 | 0.7 | Mức sáng tạo (0=ổn định, 2=sáng tạo) |
| `maxLength` | 50 - 300 | 150 | Độ dài tối đa (từ) |
| `beamSearch` | 1 - 8 | 4 | Số đường tìm kiếm (1=nhanh, 8=chất lượng) |
| `repetitionPenalty` | 1 - 2 | 1.2 | Giảm lặp lại (1=cho phép, 2=giảm mạnh) |

### Extractive (Trích xuất)

| Tham Số | Phạm vi | Mặc định | Mô tả |
|---------|--------|---------|-------|
| `extractiveSentences` | 1 - 10 | 5 | Số câu cần trích |
| `similarityThreshold` | 0 - 1 | 0.5 | Ngưỡng tương tự |

## 🎯 Cài đặt Nhanh (Presets)

### Fast Mode (Chế độ Nhanh)
```javascript
{
  temperature: 0.3,
  maxLength: 80,
  beamSearch: 1,
  repetitionPenalty: 1.0
}
```

### Balanced (Cân bằng)
```javascript
{
  temperature: 0.7,
  maxLength: 150,
  beamSearch: 4,
  repetitionPenalty: 1.2
}
```

### High Quality (Chất Lượng Cao)
```javascript
{
  temperature: 0.9,
  maxLength: 250,
  beamSearch: 8,
  repetitionPenalty: 1.5
}
```

## 💾 Lưu Trữ (localStorage)

Cài đặt được tự động lưu vào `localStorage` với key: `modelSettings`

```javascript
// Lấy cài đặt từ localStorage
const settings = JSON.parse(localStorage.getItem('modelSettings'));

// Xóa cài đặt
localStorage.removeItem('modelSettings');
```

## 📱 Responsive Design

- ✅ Mobile-first design
- ✅ Grid layout tự động điều chỉnh (1 cột trên mobile, 2+ trên desktop)
- ✅ Touch-friendly sliders
- ✅ Optimized for all screen sizes

## 🌙 Dark Mode

Hoàn toàn hỗ trợ dark mode:
- `dark:bg-gray-800` - Nền tối
- `dark:text-white` - Text sáng
- `dark:border-gray-700` - Border tối
- Sử dụng `prefers-color-scheme`

## 🎬 Animation

Sử dụng **Framer Motion** cho các hiệu ứng:
- Fade-in animations khi component mount
- Smooth slider transitions
- Button hover/tap effects
- Alert notifications

```jsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.3 }}
>
  Content
</motion.div>
```

## 🔌 Tích hợp API

Để gửi cài đặt tới backend:

```javascript
import { useModelSettings } from './hooks/useModelSettings';

function SummarizeButton() {
  const { settings } = useModelSettings();

  const handleSummarize = async (text) => {
    const response = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        settings, // Gửi toàn bộ settings
      }),
    });
    return response.json();
  };

  return <button onClick={() => handleSummarize('...')}>Tóm tắt</button>;
}
```

## 🛠️ Utility Functions

### `settingsUtils.ts`

```javascript
import {
  estimateReadingTime,
  estimateTokenCount,
  estimateProcessingTime,
  validateSettings,
  getQualityDescription,
  exportSettings,
  importSettings,
  getRecommendations,
} from './utils/settingsUtils';

// Ví dụ
const time = estimateReadingTime(150); // "~1 phút"
const tokens = estimateTokenCount(150); // 188
const procTime = estimateProcessingTime(4, 'vit5'); // "~3-5 giây"
const quality = getQualityDescription(settings); // { level: 'High', description: '...' }
const recommendations = getRecommendations('quality'); // { ... }
```

## 🧪 Testing

### Unit Tests (Jest)

```javascript
import { validateSettings, estimateReadingTime } from '../utils/settingsUtils';

test('validateSettings should return true for valid settings', () => {
  const valid = { temperature: 0.7, maxLength: 150 };
  expect(validateSettings(valid)).toBe(true);
});

test('estimateReadingTime should calculate correctly', () => {
  expect(estimateReadingTime(150)).toBe('~1 phút');
});
```

## 🚀 Performance Tips

1. **Sử dụng Context** cho app-wide settings (tránh prop drilling)
2. **Memoize components** nếu có nhiều sub-components
3. **Lazy load** heavy components khi cần
4. **Debounce** slider changes nếu có callback API

```javascript
import { useMemo } from 'react';

const memoizedComponent = useMemo(
  () => <ModelSettings key={settings.abstractiveModel} />,
  [settings.abstractiveModel]
);
```

## 📚 TypeScript Support

Tất cả các type đã được định nghĩa trong `types/modelSettings.ts`:

```typescript
import type {
  ModelSettings,
  AbstractiveSettings,
  ExtractiveSettings,
  PresetSettings,
} from '../types/modelSettings';

const settings: ModelSettings = {
  // IDE sẽ gợi ý toàn bộ fields
  temperature: 0.7,
  // ...
};
```

## 🐛 Debugging

### Enable Console Logging

```javascript
// Trong useModelSettings hook
if (process.env.DEBUG_SETTINGS) {
  console.log('Settings updated:', settings);
}
```

### DevTools

- React DevTools: Inspect context và state
- localStorage inspector: Xem cài đặt được lưu
- Network tab: Xem API calls

## 📋 Checklist Tích hợp

- [ ] Import `ModelSettings` component vào app
- [ ] Thêm `ModelSettingsProvider` nếu dùng context
- [ ] Cấu hình API endpoint để gửi settings
- [ ] Test trên mobile và dark mode
- [ ] Thêm validation rules nếu cần
- [ ] Update documentation

## 🎯 Roadmap

- [ ] Export/Import settings as JSON
- [ ] Settings profiles (save multiple presets)
- [ ] Settings comparison tool
- [ ] A/B testing mode
- [ ] Analytics tracking
- [ ] Settings sync across devices

## 📞 Support

Tham khảo:
- `types/modelSettings.ts` - Type definitions
- `utils/settingsUtils.ts` - Helper functions
- `hooks/useModelSettings.ts` - Custom hook
- `context/ModelSettingsContext.tsx` - Context provider
