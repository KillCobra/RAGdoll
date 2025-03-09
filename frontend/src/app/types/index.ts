export interface AnalysisRequest {
    description: string;
    marketSector: string;
    maxKeywords?: number;
    maxPdfLinks?: number;
  }
  
  export interface AnalysisResponse {
    text: string;
    sources: string[];
  }