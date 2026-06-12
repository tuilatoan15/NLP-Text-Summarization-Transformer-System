import React, { useState } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import {
  createAppQueryClient,
  createSessionPersister,
  SESSION_CACHE_MAX_AGE_MS,
} from '../lib/queryClient';

type Props = { children: React.ReactNode };

export function QueryProvider({ children }: Props) {
  const [queryClient] = useState(createAppQueryClient);
  const persister = createSessionPersister();

  if (!persister) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }

  return (
    <PersistQueryClientProvider
      client={queryClient}
      persistOptions={{
        persister,
        maxAge: SESSION_CACHE_MAX_AGE_MS,
        dehydrateOptions: {
          shouldDehydrateQuery: (query) => query.state.status === 'success',
        },
      }}
    >
      {children}
    </PersistQueryClientProvider>
  );
}
