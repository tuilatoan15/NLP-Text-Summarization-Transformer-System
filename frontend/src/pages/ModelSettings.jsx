import React, { useState } from 'react';
import { Save, RotateCcw, AlertCircle } from 'lucide-react';
import { useApp } from '../context/AppContext';

const ModelSettings = () => {
  const { t } = useApp();
  const [settings, setSettings] = useState({
    temperature: 0.7,
    maxLength: 150,
    topK: 50,
    topP: 0.95,
    algorithm: 'vit5',
    extractiveSentences: 5,
  });
  const [saved, setSaved] = useState(false);

  const handleChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleReset = () => {
    setSettings({
      temperature: 0.7,
      maxLength: 150,
      topK: 50,
      topP: 0.95,
      algorithm: 'vit5',
      extractiveSentences: 5,
    });
  };

  return (
    <div className="space-y-6 pb-12 max-w-3xl">
      <div>
        <h1 className="ui-page-title mb-1">{t('settingsTitle')}</h1>
        <p className="ui-page-subtitle">{t('settingsSubtitle')}</p>
      </div>

      <div className="rounded-xl border border-blue-200 dark:border-blue-800/60 bg-blue-50 dark:bg-blue-950/30 p-4 flex gap-3">
        <AlertCircle className="text-blue-600 dark:text-blue-400 flex-shrink-0" size={20} />
        <div className="text-sm text-blue-800 dark:text-blue-200">
          <strong>{t('settingsNote')}</strong> {t('settingsNoteBody')}
        </div>
      </div>

      <div className="space-y-6">
        <div className="ui-card p-6">
          <h2 className="text-lg font-semibold text-[var(--text)] mb-6">{t('settingsAbstractive')}</h2>

          <div className="space-y-6">
            <div>
              <label className="ui-label mb-2">{t('settingsSelectModel')}</label>
              <select
                value={settings.algorithm}
                onChange={(e) => handleChange('algorithm', e.target.value)}
                className="ui-input py-3"
              >
                <option value="vit5">ViT5 (Fine-tuned)</option>
                <option value="mt5">mT5 (Multilingual)</option>
                <option value="bartpho">BARTPho (Vietnamese)</option>
              </select>
              <p className="text-xs text-[var(--text-muted)] mt-2">{t('settingsModelHint')}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="ui-label mb-3">
                  {t('settingsTemperature')}
                  <span className="float-right text-lg font-bold text-blue-600 dark:text-blue-400">
                    {settings.temperature.toFixed(2)}
                  </span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                  className="ui-range bg-[var(--surface-inset)]"
                />
                <p className="text-xs text-[var(--text-muted)] mt-2">{t('settingsTemperatureHint')}</p>
              </div>

              <div>
                <label className="ui-label mb-3">
                  {t('settingsMaxLength')}
                  <span className="float-right text-lg font-bold text-blue-600 dark:text-blue-400">
                    {settings.maxLength}
                  </span>
                </label>
                <input
                  type="range"
                  min="50"
                  max="512"
                  step="10"
                  value={settings.maxLength}
                  onChange={(e) => handleChange('maxLength', parseInt(e.target.value, 10))}
                  className="ui-range bg-[var(--surface-inset)]"
                />
                <p className="text-xs text-[var(--text-muted)] mt-2">{t('settingsMaxLengthHint')}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="ui-label mb-3">
                  {t('settingsTopK')}
                  <span className="float-right text-lg font-bold text-blue-600 dark:text-blue-400">
                    {settings.topK}
                  </span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="5"
                  value={settings.topK}
                  onChange={(e) => handleChange('topK', parseInt(e.target.value, 10))}
                  className="ui-range bg-[var(--surface-inset)]"
                />
                <p className="text-xs text-[var(--text-muted)] mt-2">{t('settingsTopKHint')}</p>
              </div>

              <div>
                <label className="ui-label mb-3">
                  {t('settingsTopP')}
                  <span className="float-right text-lg font-bold text-blue-600 dark:text-blue-400">
                    {settings.topP.toFixed(2)}
                  </span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={settings.topP}
                  onChange={(e) => handleChange('topP', parseFloat(e.target.value))}
                  className="ui-range bg-[var(--surface-inset)]"
                />
                <p className="text-xs text-[var(--text-muted)] mt-2">{t('settingsTopPHint')}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="ui-card p-6">
          <h2 className="text-lg font-semibold text-[var(--text)] mb-6">{t('settingsExtractive')}</h2>

          <div className="space-y-6">
            <div>
              <label className="ui-label mb-3">
                {t('settingsSentences')}
                <span className="float-right text-lg font-bold text-blue-600 dark:text-blue-400">
                  {settings.extractiveSentences}
                </span>
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={settings.extractiveSentences}
                onChange={(e) => handleChange('extractiveSentences', parseInt(e.target.value, 10))}
                className="ui-range bg-[var(--surface-inset)]"
              />
              <p className="text-xs text-[var(--text-muted)] mt-2">{t('settingsSentencesHint')}</p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {['TextRank', 'LexRank', 'LSA'].map(algo => (
                <button
                  key={algo}
                  type="button"
                  className="px-4 py-3 border-2 rounded-lg text-sm font-medium transition border-[var(--border)] bg-[var(--surface-inset)] text-[var(--text-secondary)] hover:border-blue-500 dark:hover:border-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
                >
                  {algo}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex gap-3">
          <button type="button" onClick={handleSave} className="ui-btn-primary flex-1">
            <Save size={18} />
            {t('settingsSave')}
          </button>
          <button type="button" onClick={handleReset} className="ui-btn-secondary flex-1">
            <RotateCcw size={18} />
            {t('settingsReset')}
          </button>
        </div>

        {saved && (
          <div className="rounded-lg border border-emerald-200 dark:border-emerald-800/50 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-800 dark:text-emerald-200 text-sm p-4">
            ✓ {t('settingsSaved')}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelSettings;
