'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import LoadingSpinner from './LoadingSpinner';
import styles from './AnalysisForm.module.css';

export default function AnalysisForm() {
  const [formData, setFormData] = useState({
    description: '',
    market_or_sector: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze');
      }

      sessionStorage.setItem('analysisResults', JSON.stringify(data));
      router.push('/results');
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.formWrapper}>
      <form onSubmit={handleSubmit} className={styles.form}>
        {error && (
          <div className={styles.error}>
            {error}
          </div>
        )}

        <div className={styles.formGroup}>
          <label className={styles.label}>
            Company Description
            <span className={styles.required}>(Required)</span>
          </label>
          <textarea
            required
            className={styles.textarea}
            rows={6}
            placeholder="Describe your company's core business, current market position, and goals..."
            value={formData.description}
            onChange={(e) => setFormData({...formData, description: e.target.value})}
          />
        </div>

        <div className={styles.formGroup}>
          <label className={styles.label}>
            Target Market/Sector
            <span className={styles.required}>(Required)</span>
          </label>
          <input
            type="text"
            required
            className={styles.input}
            placeholder="e.g., Rural Area, Urban Market, Technology Sector..."
            value={formData.market_or_sector}
            onChange={(e) => setFormData({...formData, market_or_sector: e.target.value})}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className={styles.button}
        >
          {isLoading ? (
            <div className={styles.loadingWrapper}>
              <LoadingSpinner size="small" color="white" text="" />
              <span>Analyzing...</span>
            </div>
          ) : (
            'Generate Risk Analysis'
          )}
        </button>
      </form>
    </div>
  );
}