import { apiClient } from './api';

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}

export interface User {
  username: string;
  email?: string;
  id?: string;
}

class AuthService {
  private static instance: AuthService;
  
  private constructor() {}
  
  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }
  
  async login(credentials: LoginCredentials): Promise<AuthToken> {
    // OAuth2 compatible login using form data
    const formData = new URLSearchParams();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    const response = await apiClient.post<AuthToken>('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    
    // Store token and user info
    if (response.access_token) {
      this.setToken(response.access_token);
      // Store basic user info
      this.setUser({ username: credentials.username });
    }
    
    return response;
  }
  
  logout(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
  }
  
  getToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('access_token');
    }
    return null;
  }
  
  setToken(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('access_token', token);
    }
  }
  
  getUser(): User | null {
    if (typeof window !== 'undefined') {
      const userStr = localStorage.getItem('user');
      if (userStr) {
        try {
          return JSON.parse(userStr);
        } catch {
          return null;
        }
      }
    }
    return null;
  }
  
  setUser(user: User): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('user', JSON.stringify(user));
    }
  }
  
  isAuthenticated(): boolean {
    return !!this.getToken();
  }
  
  async validateToken(): Promise<boolean> {
    try {
      // Call protected endpoint to validate token
      await apiClient.get('/auth/protected-route');
      return true;
    } catch {
      return false;
    }
  }
}

export const authService = AuthService.getInstance();