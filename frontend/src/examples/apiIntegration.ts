/**
 * API Integration Examples for Model Settings
 * Shows how to use the settings system with backend API calls
 */

import type { ModelSettings } from '../types/modelSettings';

// ============================================================================
// EXAMPLE 1: Direct API Call with Settings
// ============================================================================

export const summarizeWithSettings = async (
  text: string,
  settings: ModelSettings
): Promise<{ abstractive: string; extractive: string }> => {
  const response = await fetch('/api/summarize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      abstractive: {
        model: settings.abstractiveModel,
        temperature: settings.temperature,
        maxLength: settings.maxLength,
        beamSearch: settings.beamSearch,
        repetitionPenalty: settings.repetitionPenalty,
      },
      extractive: {
        algorithm: settings.extractiveAlgorithm,
        numSentences: settings.extractiveSentences,
        similarityThreshold: settings.similarityThreshold,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
};

// ============================================================================
// EXAMPLE 2: React Hook for Summarization with Settings
// ============================================================================

import { useState, useCallback } from 'react';

interface SummarizationState {
  loading: boolean;
  result: { abstractive: string; extractive: string } | null;
  error: string | null;
}

export const useSummarizeWithSettings = () => {
  const [state, setState] = useState<SummarizationState>({
    loading: false,
    result: null,
    error: null,
  });

  const summarize = useCallback(
    async (text: string, settings: ModelSettings) => {
      setState({ loading: true, result: null, error: null });

      try {
        const result = await summarizeWithSettings(text, settings);
        setState({ loading: false, result, error: null });
        return result;
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Unknown error occurred';
        setState({ loading: false, result: null, error: errorMessage });
        throw error;
      }
    },
    []
  );

  return { ...state, summarize };
};

// ============================================================================
// EXAMPLE 3: Component Usage
// ============================================================================

/**
 * Example of using settings in a component
 */
export const SummarizePlayground = () => {
  const { settings } = useModelSettings();
  const { loading, result, error, summarize } = useSummarizeWithSettings();
  const [text, setText] = useState('');

  const handleSummarize = async () => {
    try {
      await summarize(text, settings);
    } catch (err) {
      console.error('Summarization failed:', err);
    }
  };

  return (
    <div className="space-y-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Nhập văn bản cần tóm tắt..."
        className="w-full h-40 p-4 border rounded-lg"
      />

      <div className="text-sm text-gray-600">
        📊 Sử dụng:
        <ul className="mt-2 space-y-1">
          <li>• Model: {settings.abstractiveModel}</li>
          <li>• Temperature: {settings.temperature}</li>
          <li>• Max Length: {settings.maxLength}</li>
          <li>• Beam Search: {settings.beamSearch}</li>
        </ul>
      </div>

      <button
        onClick={handleSummarize}
        disabled={loading}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? 'Đang tóm tắt...' : 'Tóm tắt'}
      </button>

      {error && <div className="p-4 bg-red-50 text-red-700 rounded-lg">{error}</div>}

      {result && (
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold mb-2">Abstractive</h3>
            <p className="text-sm">{result.abstractive}</p>
          </div>
          <div className="p-4 bg-green-50 rounded-lg">
            <h3 className="font-semibold mb-2">Extractive</h3>
            <p className="text-sm">{result.extractive}</p>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// EXAMPLE 4: Batch Processing with Settings
// ============================================================================

interface BatchItem {
  id: string;
  text: string;
}

export const batchSummarizeWithSettings = async (
  items: BatchItem[],
  settings: ModelSettings,
  onProgress?: (current: number, total: number) => void
): Promise<Record<string, { abstractive: string; extractive: string }>> => {
  const results: Record<string, { abstractive: string; extractive: string }> = {};

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    try {
      results[item.id] = await summarizeWithSettings(item.text, settings);
      onProgress?.(i + 1, items.length);
    } catch (error) {
      console.error(`Failed to summarize item ${item.id}:`, error);
      results[item.id] = {
        abstractive: 'Error',
        extractive: 'Error',
      };
    }
  }

  return results;
};

// ============================================================================
// EXAMPLE 5: Settings Validation Before API Call
// ============================================================================

import { validateSettings } from '../utils/settingsUtils';

export const summarizeWithValidation = async (
  text: string,
  settings: ModelSettings
): Promise<{ abstractive: string; extractive: string }> => {
  // Validate settings first
  if (!validateSettings(settings)) {
    throw new Error('Invalid settings provided');
  }

  // Additional custom validation
  if (text.trim().length === 0) {
    throw new Error('Text cannot be empty');
  }

  if (text.length < 50) {
    console.warn('Warning: Text is very short, summarization may not be effective');
  }

  // Proceed with API call
  return summarizeWithSettings(text, settings);
};

// ============================================================================
// EXAMPLE 6: Settings Comparison
// ============================================================================

export const compareSettings = async (
  text: string,
  settingsA: ModelSettings,
  settingsB: ModelSettings
) => {
  const [resultA, resultB] = await Promise.all([
    summarizeWithSettings(text, settingsA),
    summarizeWithSettings(text, settingsB),
  ]);

  return {
    settings_a: {
      config: settingsA,
      result: resultA,
    },
    settings_b: {
      config: settingsB,
      result: resultB,
    },
  };
};

// ============================================================================
// EXAMPLE 7: Settings with Caching
// ============================================================================

interface CacheEntry {
  timestamp: number;
  result: { abstractive: string; extractive: string };
}

class SettingsCachingService {
  private cache: Map<string, CacheEntry> = new Map();
  private cacheTTL = 5 * 60 * 1000; // 5 minutes

  generateKey(text: string, settings: ModelSettings): string {
    return `${text}_${JSON.stringify(settings)}`;
  }

  async summarize(
    text: string,
    settings: ModelSettings
  ): Promise<{ abstractive: string; extractive: string }> {
    const key = this.generateKey(text, settings);

    // Check cache
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      console.log('✓ Returning cached result');
      return cached.result;
    }

    // API call
    const result = await summarizeWithSettings(text, settings);

    // Store in cache
    this.cache.set(key, {
      timestamp: Date.now(),
      result,
    });

    return result;
  }

  clearCache(): void {
    this.cache.clear();
  }
}

export const cachingService = new SettingsCachingService();

// ============================================================================
// EXAMPLE 8: Environment-specific Settings
// ============================================================================

export const getEnvironmentSettings = (): Partial<ModelSettings> => {
  const env = process.env.NODE_ENV;

  if (env === 'development') {
    return {
      beamSearch: 2, // Faster for development
      temperature: 0.5,
    };
  }

  if (env === 'production') {
    return {
      beamSearch: 6, // Better quality for production
      temperature: 0.7,
    };
  }

  return {};
};
