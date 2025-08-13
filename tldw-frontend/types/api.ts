// API Response Types

export interface MediaItem {
  id: number;
  title: string;
  author?: string;
  content?: string;
  description?: string;
  url?: string;
  type: 'video' | 'audio' | 'document' | 'article' | 'podcast';
  transcription?: string;
  summary?: string;
  created_at: string;
  updated_at: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface MediaListResponse {
  items: MediaItem[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface SearchResult {
  media_id: number;
  title: string;
  content_snippet: string;
  relevance_score: number;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  search_type: 'title' | 'content' | 'both';
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

export interface ChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface Note {
  id: number;
  title: string;
  content: string;
  tags?: string[];
  created_at: string;
  updated_at: string;
}

export interface Prompt {
  id: number;
  name: string;
  content: string;
  description?: string;
  category?: string;
  tags?: string[];
  created_at: string;
  updated_at: string;
}

export interface TranscriptionRequest {
  file?: File;
  url?: string;
  language?: string;
  model?: string;
}

export interface TranscriptionResponse {
  text: string;
  language?: string;
  duration?: number;
  segments?: Array<{
    start: number;
    end: number;
    text: string;
  }>;
}

export interface ErrorResponse {
  detail: string;
  status_code?: number;
  type?: string;
}

// Pagination params
export interface PaginationParams {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

// Filter params
export interface FilterParams {
  type?: string;
  tags?: string[];
  date_from?: string;
  date_to?: string;
  author?: string;
}