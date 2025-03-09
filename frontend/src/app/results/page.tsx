'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

interface AnalysisResult {
  text: string;
  sources: string[];
}

export default function ResultsPage() {
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const stored = sessionStorage.getItem('analysisResults');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setResults({
          text: parsed.text || parsed.message || '',
          sources: Array.isArray(parsed.sources) ? parsed.sources : []
        });
      } catch (e) {
        setError('Failed to parse results');
        console.error('Error parsing results:', e);
      }
    }
  }, []);

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{error}</p>
          <button
            onClick={() => router.push('/')}
            className="mt-4 bg-red-500 text-white px-4 py-2 rounded"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Risk Analysis Results</h1>
      <div className="bg-white shadow rounded-lg p-6">
        <div className="prose max-w-none">
          <h2 className="text-xl font-semibold mb-4">Analysis</h2>
          <div className="whitespace-pre-wrap">{results.text}</div>
          
          {results.sources && results.sources.length > 0 && (
            <>
              <h3 className="text-lg font-semibold mt-6 mb-2">Sources</h3>
              <ul className="list-disc pl-5">
                {results.sources.map((source, index) => (
                  <li key={index} className="text-sm text-gray-600">{source}</li>
                ))}
              </ul>
            </>
          )}
        </div>
        
        <button
          onClick={() => router.push('/')}
          className="mt-8 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
        >
          New Analysis
        </button>
      </div>
    </div>
  );
}