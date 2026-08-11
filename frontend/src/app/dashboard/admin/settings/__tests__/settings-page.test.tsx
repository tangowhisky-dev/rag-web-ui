/**
 * Tests for the Super Admin Settings page.
 *
 * Verifies:
 *  1. Renders settings grouped by category
 *  2. Shows save button as disabled when no changes
 *  3. Shows unsaved indicator when a field is modified
 *  4. Calls the correct API endpoint on save
 *  5. Displays reindex warnings for settings that require it
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: jest.fn() }),
}));

// Mock the API
const mockApiGet = jest.fn();
const mockApiPut = jest.fn();
const mockApiDelete = jest.fn();
jest.mock('@/lib/api', () => ({
  api: {
    get: (...args: any[]) => mockApiGet(...args),
    put: (...args: any[]) => mockApiPut(...args),
    delete: (...args: any[]) => mockApiDelete(...args),
  },
}));

// Mock useToast
const mockToast = jest.fn();
jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({ toast: mockToast }),
}));

// Mock Select component to avoid Radix portal complexity
jest.mock('@/components/ui/select', () => ({
  Select: ({ children, value, onValueChange }: any) => (
    <select value={value} onChange={(e) => onValueChange(e.target.value)}>{children}</select>
  ),
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
  SelectTrigger: ({ children }: any) => <>{children}</>,
  SelectValue: ({ placeholder }: any) => <option value="">{placeholder}</option>,
}));

// Mock ConfirmDialog
jest.mock('@/components/ui/confirm-dialog', () => ({
  ConfirmDialog: ({ open, title, onConfirm, onCancel }: any) =>
    open ? (
      <div>
        <p>{title}</p>
        <button onClick={onConfirm}>Confirm</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    ) : null,
}));

import SuperAdminSettingsPage from '../page';

const MOCK_SETTINGS = {
  settings: [
    {
      key: 'RETRIEVAL_TOP_K',
      value: 20,
      value_type: 'int',
      category: 'Retrieval',
      label: 'Top-K',
      scope: 'org',
      source: 'install_default',
      reload: 'next_request',
      requires_reindex: false,
      description: 'Number of chunks to retrieve',
      min: 1,
      max: 200,
      choices: null,
    },
    {
      key: 'CHUNK_SIZE',
      value: 1500,
      value_type: 'int',
      category: 'Ingestion',
      label: 'Chunk size',
      scope: 'app',
      source: 'install_default',
      reload: 'ingest',
      requires_reindex: true,
      description: 'DataStores are shared across orgs',
      min: 100,
      max: 8000,
      choices: null,
    },
    {
      key: 'RERANKER_ENABLED',
      value: true,
      value_type: 'bool',
      category: 'Reranker',
      label: 'Enable reranker',
      scope: 'org',
      source: 'install_default',
      reload: 'next_request',
      requires_reindex: false,
      description: 'Cross-encoder reranking',
      min: null,
      max: null,
      choices: null,
    },
  ],
};

describe('SuperAdminSettingsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiGet.mockResolvedValue(MOCK_SETTINGS);
    mockApiPut.mockResolvedValue({ results: [] });
  });

  it('renders settings grouped by category', async () => {
    render(<SuperAdminSettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Retrieval')).toBeInTheDocument();
      expect(screen.getByText('Ingestion')).toBeInTheDocument();
      expect(screen.getByText('Reranker')).toBeInTheDocument();
    });
  });

  it('shows save button as disabled when no changes', async () => {
    render(<SuperAdminSettingsPage />);

    await waitFor(() => {
      const saveButton = screen.getByText('Save');
      expect(saveButton).toBeDisabled();
    });
  });

  it('enables save button after a field is modified', async () => {
    render(<SuperAdminSettingsPage />);

    await waitFor(() => {
      expect(screen.getByDisplayValue('20')).toBeInTheDocument();
    });

    const input = screen.getByDisplayValue('20');
    fireEvent.change(input, { target: { value: '30' } });

    await waitFor(() => {
      const saveButton = screen.getByText('Save');
      expect(saveButton).not.toBeDisabled();
    });
  });

  it('calls the correct API endpoint on save', async () => {
    render(<SuperAdminSettingsPage />);

    await waitFor(() => {
      expect(screen.getByDisplayValue('20')).toBeInTheDocument();
    });

    const input = screen.getByDisplayValue('20');
    fireEvent.change(input, { target: { value: '30' } });

    const saveButton = screen.getByText('Save');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockApiPut).toHaveBeenCalledWith('/api/admin/settings', {
        settings: [{ key: 'RETRIEVAL_TOP_K', value: 30 }],
      });
    });
  });

  it('displays reindex warning for settings that require it', async () => {
    render(<SuperAdminSettingsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Requires re-indexing/i)).toBeInTheDocument();
    });
  });

  it('shows org-overridable badge for org-scoped settings', async () => {
    const { container } = render(<SuperAdminSettingsPage />);

    await waitFor(() => {
      expect(container.textContent).toContain('org-overridable');
    });
  });
});
