interface FetchOptions extends Omit<RequestInit, 'body' | 'headers'> {
  data?: unknown;
  headers?: Record<string, string>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public headers?: Headers,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function buildConfig(options: FetchOptions): RequestInit {
  const { data, headers: customHeaders = {}, ...restOptions } = options;

  const headers: Record<string, string> = {
    ...customHeaders,
  };

  // If no content type is specified and we have data, default to JSON
  if (!headers['Content-Type'] && data && !(data instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const config: RequestInit = {
    ...restOptions,
    credentials: 'include',
    headers,
  };

  // Handle body based on Content-Type
  if (data) {
    if (data instanceof FormData) {
      config.body = data;
    } else if (headers['Content-Type'] === 'application/json') {
      config.body = JSON.stringify(data);
    } else if (headers['Content-Type'] === 'application/x-www-form-urlencoded') {
      config.body = typeof data === 'string' ? data : new URLSearchParams(data as Record<string, string>).toString();
    } else {
      config.body = data as BodyInit;
    }
  }

  return config;
}

/** Fetch with auth + error handling, returning the raw Response (for blob/stream cases). */
export async function fetchRaw(fullUrl: string, options: FetchOptions = {}): Promise<Response> {
  const config = buildConfig(options);
  try {
    const response = await fetch(fullUrl, config);

    // Skip 401 redirect for login/logout endpoints
    if (response.status === 401 && !fullUrl.includes('/api/auth/token') && !fullUrl.includes('/api/auth/logout')) {
      if (typeof window !== 'undefined') {
        try {
          await api.post('/api/auth/logout');
        } catch {
          // best effort
        }
        window.location.href = '/';
      }
      throw new ApiError(401, 'Unauthorized - Please log in again');
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        response.status,
        errorData.message || errorData.detail || 'An error occurred',
        response.headers,
      );
    }

    return response;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(500, 'Network error or server is unreachable');
  }
}

/** Fetch with auth + error handling, parsing the response as JSON. */
export async function fetchApi(fullUrl: string, options: FetchOptions = {}): Promise<any> {
  const response = await fetchRaw(fullUrl, options);

  // Handle 204 No Content - return null instead of trying to parse JSON
  if (response.status === 204) {
    return null;
  }

  return await response.json();
}

// Helper methods for common HTTP methods
export const api = {
  get: (url: string, options?: Omit<FetchOptions, 'method'>) =>
    fetchApi(url, { ...options, method: 'GET' }),

  getRaw: (url: string, options?: Omit<FetchOptions, 'method'>) =>
    fetchRaw(url, { ...options, method: 'GET' }),

  post: (url: string, data?: unknown, options?: Omit<FetchOptions, 'method'>) =>
    fetchApi(url, { ...options, method: 'POST', data }),

  postRaw: (url: string, data?: unknown, options?: Omit<FetchOptions, 'method'>) =>
    fetchRaw(url, { ...options, method: 'POST', data }),

  put: (url: string, data?: unknown, options?: Omit<FetchOptions, 'method'>) =>
    fetchApi(url, { ...options, method: 'PUT', data }),

  delete: (url: string, options?: Omit<FetchOptions, 'method'>) =>
    fetchApi(url, { ...options, method: 'DELETE' }),

  patch: (url: string, data?: unknown, options?: Omit<FetchOptions, 'method'>) =>
    fetchApi(url, { ...options, method: 'PATCH', data }),
};
