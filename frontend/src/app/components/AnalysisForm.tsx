'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import LoadingSpinner from './LoadingSpinner';

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
    <form onSubmit={handleSubmit} className="space-y-6">
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      <div>
        <label className="block text-sm font-medium mb-2">
          Company Description
        </label>
        <textarea
          required
          className="w-full p-2 border rounded"
          rows={4}
          value={formData.description}
          onChange={(e) => setFormData({...formData, description: e.target.value})}
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">
          Market/Sector
        </label>
        <input
          type="text"
          required
          className="w-full p-2 border rounded"
          value={formData.market_or_sector}
          onChange={(e) => setFormData({...formData, market_or_sector: e.target.value})}
        />
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className={`w-full py-2 px-4 rounded text-white ${
          isLoading ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'
        }`}
      >
        {isLoading ? <LoadingSpinner /> : 'Submit for Analysis'}
      </button>
    </form>
  );
}