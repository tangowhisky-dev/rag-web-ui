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

// Mock Tabs to render all content (avoid Radix portal/visibility issues)
jest.mock('@/components/ui/tabs', () => ({
  Tabs: ({ children }: any) => <div>{children}</div>,
  TabsList: ({ children }: any) => <div>{children}</div>,
  TabsTrigger: ({ children, value }: any) => <div data-tab={value}>{children}</div>,
  TabsContent: ({ children }: any) => <div>{children}</div>,
}));

// Mock ModelPicker to avoid network calls
jest.mock('@/components/settings/model-picker', () => ({
  ModelPicker: ({ value, onChange, placeholder }: any) => (
    <input value={value ?? ''} onChange={(e) => onChange(e.target.value || null)} placeholder={placeholder} data-testid="model-picker" />
  ),
}));

import SuperAdminSettingsPage from '../page';

const MOCK_SETTINGS = {
  settings: [
    {
      key: 'RETRIEVAL_TOP_K',
      value: 20,
      default: 20,
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
      model_picker: false,
      api_base_ref: null,
      api_key_ref: null,
    },
    {
      key: 'CHUNK_SIZE',
      value: 1500,
      default: 1500,
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
      model_picker: false,
      api_base_ref: null,
      api_key_ref: null,
    },
    {
      key: 'RERANKER_SCORE_THRESHOLD',
      value: -2.0,
      default: -2.0,
      value_type: 'float',
      category: 'Reranker',
      label: 'Reranker score threshold',
      scope: 'org',
      source: 'install_default',
      reload: 'next_request',
      requires_reindex: false,
      description: 'Minimum cross-encoder logit to pass reranking',
      min: null,
      max: null,
      choices: null,
      model_picker: false,
      api_base_ref: null,
      api_key_ref: null,
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
      // Category headers appear in tab content; tab labels may also match.
      // Use getAllByText to assert at least one element renders for each.
      expect(screen.getAllByText('Retrieval').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Ingestion').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Reranker').length).toBeGreaterThan(0);
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
