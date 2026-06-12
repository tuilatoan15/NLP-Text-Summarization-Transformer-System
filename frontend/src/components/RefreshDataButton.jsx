import React, { useCallback, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useApp } from '../context/AppContext';
import { invalidateAllCaches } from '../lib/cacheInvalidation';
import { formatCacheSize, estimateCacheSizeBytes } from '../lib/cacheLogger';

export default function RefreshDataButton() {
  const { t } = useApp();
  const queryClient = useQueryClient();
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await invalidateAllCaches(queryClient);
      await queryClient.refetchQueries({ type: 'active' });
    } finally {
      setRefreshing(false);
    }
  }, [queryClient]);

  return (
    <button
      type="button"
      onClick={handleRefresh}
      disabled={refreshing}
      title={`${t('refreshData', 'Làm mới dữ liệu')} · ${formatCacheSize(estimateCacheSizeBytes())}`}
      className="ui-btn-icon cursor-pointer hidden md:inline-flex"
    >
      <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
    </button>
  );
}
