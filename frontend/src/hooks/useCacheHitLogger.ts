import { useEffect, useRef } from 'react';
import { cacheLog } from '../lib/cacheLogger';

export function useCacheHitLogger(label: string, data: unknown, isFetching: boolean): void {
  const logged = useRef(false);
  useEffect(() => {
    if (data !== undefined && data !== null && !isFetching && !logged.current) {
      logged.current = true;
      cacheLog('HIT', label, 'in-memory');
    }
  }, [label, data, isFetching]);
}
