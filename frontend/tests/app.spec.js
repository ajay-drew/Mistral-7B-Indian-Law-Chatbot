import { test, expect } from '@playwright/test';

test.describe('Mistral Indian Law Frontend', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API responses
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ documents: [] }),
      });
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should display the app header with title', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Mistral Indian Law');
    await expect(page.locator('.subtitle')).toContainText('Your AI Assistant for Indian Legal Matters');
  });

  test('should display created by section', async ({ page }) => {
    const createdBy = page.locator('.created-by');
    await expect(createdBy).toBeVisible();
    await expect(createdBy).toContainText('Created by');
    await expect(createdBy).toContainText('Ajay A');
    await expect(createdBy).toContainText('drewjay05@gmail.com');
  });

  test('should show welcome message when no documents are uploaded', async ({ page }) => {
    await expect(page.locator('.welcome-message')).toBeVisible();
    await expect(page.locator('.welcome-message h2')).toContainText('Upload Documents to Get Started');
  });

  test('should have document upload section', async ({ page }) => {
    await expect(page.locator('.document-upload h3')).toContainText('Upload Documents');
    await expect(page.locator('.document-hint')).toContainText('Upload PDF documents to enable chat functionality');
  });

  test('should have theme toggle button', async ({ page }) => {
    const themeToggle = page.locator('.theme-toggle');
    await expect(themeToggle).toBeVisible();
    await expect(themeToggle).toHaveAttribute('aria-label', 'Toggle theme');
  });

  test('should toggle theme when theme button is clicked', async ({ page }) => {
    const themeToggle = page.locator('.theme-toggle');
    const html = page.locator('html');
    
    // Check initial theme (should be light by default)
    await expect(html).toHaveAttribute('data-theme', 'light');
    
    // Click theme toggle
    await themeToggle.click();
    await page.waitForTimeout(500); // Wait for theme transition
    
    // Check theme changed to dark
    await expect(html).toHaveAttribute('data-theme', 'dark');
    
    // Click again to toggle back
    await themeToggle.click();
    await page.waitForTimeout(500);
    
    // Check theme changed back to light
    await expect(html).toHaveAttribute('data-theme', 'light');
  });

  test('should disable chat input when no documents are uploaded', async ({ page }) => {
    const chatInput = page.locator('.input-field');
    const sendButton = page.locator('.send-button');
    
    await expect(chatInput).toBeDisabled();
    await expect(sendButton).toBeDisabled();
    await expect(chatInput).toHaveAttribute('placeholder', 'Upload a document to start chatting...');
  });

  test('should show upload error for non-PDF files', async ({ page }) => {
    // Mock file upload endpoint
    await page.route('**/documents/upload', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Please upload a PDF file' }),
      });
    });

    // Create a fake file input
    const fileInput = page.locator('#file-upload');
    
    // Try to upload a non-PDF file (simulated)
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('test content'),
    });

    // Wait for error message
    await expect(page.locator('.upload-error')).toBeVisible({ timeout: 5000 });
  });

  test('should handle document upload successfully', async ({ page }) => {
    // Mock successful upload
    await page.route('**/documents/upload', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'test-doc-1',
          filename: 'test.pdf',
          file_size: 1024,
          chunk_count: 5,
          upload_date: new Date().toISOString(),
        }),
      });
    });

    // Mock documents list after upload
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [{
            id: 'test-doc-1',
            filename: 'test.pdf',
            file_size: 1024,
            chunk_count: 5,
            upload_date: new Date().toISOString(),
          }],
        }),
      });
    });

    // Create a fake PDF file
    const fileInput = page.locator('#file-upload');
    await fileInput.setInputFiles({
      name: 'test.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('%PDF-1.4 fake pdf content'),
    });

    // Wait for document to appear in list
    await expect(page.locator('.document-list')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('.document-name')).toContainText('test.pdf');
  });

  test('should display uploaded documents', async ({ page }) => {
    // Mock documents list with documents
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              id: 'doc-1',
              filename: 'document1.pdf',
              file_size: 2048,
              chunk_count: 10,
              upload_date: new Date().toISOString(),
            },
            {
              id: 'doc-2',
              filename: 'document2.pdf',
              file_size: 4096,
              chunk_count: 20,
              upload_date: new Date().toISOString(),
            },
          ],
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    await expect(page.locator('.document-list')).toBeVisible();
    await expect(page.locator('.document-card')).toHaveCount(2);
    await expect(page.locator('.document-name').first()).toContainText('document1.pdf');
    await expect(page.locator('.document-name').last()).toContainText('document2.pdf');
  });

  test('should enable chat when documents are uploaded', async ({ page }) => {
    // Mock documents list with a document
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [{
            id: 'doc-1',
            filename: 'test.pdf',
            file_size: 1024,
            chunk_count: 5,
            upload_date: new Date().toISOString(),
          }],
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    const chatInput = page.locator('.input-field');
    const sendButton = page.locator('.send-button');
    
    await expect(chatInput).not.toBeDisabled();
    await expect(chatInput).toHaveAttribute('placeholder', 'Ask about Indian law...');
    await expect(sendButton).not.toBeDisabled();
  });

  test('should send chat message when documents are available', async ({ page }) => {
    // Mock documents list
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [{
            id: 'doc-1',
            filename: 'test.pdf',
            file_size: 1024,
            chunk_count: 5,
            upload_date: new Date().toISOString(),
          }],
        }),
      });
    });

    // Mock chat endpoint
    await page.route('**/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          reply: 'This is a test response from the AI assistant.',
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    const chatInput = page.locator('.input-field');
    const sendButton = page.locator('.send-button');

    await chatInput.fill('What is Indian law?');
    await sendButton.click();

    // Wait for user message to appear
    await expect(page.locator('.user-message')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('.user-message .message-text')).toContainText('What is Indian law?');

    // Wait for assistant response
    await expect(page.locator('.assistant-message').last()).toBeVisible({ timeout: 10000 });
    await expect(page.locator('.assistant-message .message-text').last()).toContainText('This is a test response');
  });

  test('should show error message when chat request fails', async ({ page }) => {
    // Mock documents list
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [{
            id: 'doc-1',
            filename: 'test.pdf',
            file_size: 1024,
            chunk_count: 5,
            upload_date: new Date().toISOString(),
          }],
        }),
      });
    });

    // Mock chat endpoint to fail
    await page.route('**/chat', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Internal server error',
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    const chatInput = page.locator('.input-field');
    const sendButton = page.locator('.send-button');

    await chatInput.fill('Test question');
    await sendButton.click();

    // Wait for error message
    await expect(page.locator('.assistant-message').last()).toBeVisible({ timeout: 10000 });
    await expect(page.locator('.assistant-message .message-text').last()).toContainText('error');
  });

  test('should delete document when delete button is clicked', async ({ page }) => {
    // Mock initial documents list
    await page.route('**/documents', async (route) => {
      const url = route.request().url();
      if (url.includes('/documents/doc-1')) {
        // Delete endpoint
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Document deleted' }),
        });
      } else {
        // List endpoint - return one document initially
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            documents: [{
              id: 'doc-1',
              filename: 'test.pdf',
              file_size: 1024,
              chunk_count: 5,
              upload_date: new Date().toISOString(),
            }],
          }),
        });
      }
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    // Mock documents list to return empty after delete
    await page.route('**/documents', async (route) => {
      const url = route.request().url();
      if (url.includes('/documents/doc-1')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Document deleted' }),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ documents: [] }),
        });
      }
    });

    const deleteButton = page.locator('.delete-button');
    await deleteButton.click();

    // Wait for document to be removed
    await expect(page.locator('.document-list')).not.toBeVisible({ timeout: 5000 });
  });

  test('should show loading indicator while sending message', async ({ page }) => {
    // Mock documents list
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [{
            id: 'doc-1',
            filename: 'test.pdf',
            file_size: 1024,
            chunk_count: 5,
            upload_date: new Date().toISOString(),
          }],
        }),
      });
    });

    // Mock chat endpoint with delay
    await page.route('**/chat', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          reply: 'Response',
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    const chatInput = page.locator('.input-field');
    const sendButton = page.locator('.send-button');

    await chatInput.fill('Test question');
    await sendButton.click();

    // Check for loading indicator
    await expect(page.locator('.loading-dots')).toBeVisible({ timeout: 2000 });
  });

  test('should handle Enter key to send message', async ({ page }) => {
    // Mock documents list
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [{
            id: 'doc-1',
            filename: 'test.pdf',
            file_size: 1024,
            chunk_count: 5,
            upload_date: new Date().toISOString(),
          }],
        }),
      });
    });

    // Mock chat endpoint
    await page.route('**/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          reply: 'Response',
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    const chatInput = page.locator('.input-field');
    await chatInput.fill('Test question');
    await chatInput.press('Enter');

    // Wait for message to appear
    await expect(page.locator('.user-message')).toBeVisible({ timeout: 5000 });
  });

  test('should show document badge when documents are loaded', async ({ page }) => {
    // Mock documents list with documents
    await page.route('**/documents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              id: 'doc-1',
              filename: 'test1.pdf',
              file_size: 1024,
              chunk_count: 5,
              upload_date: new Date().toISOString(),
            },
            {
              id: 'doc-2',
              filename: 'test2.pdf',
              file_size: 2048,
              chunk_count: 10,
              upload_date: new Date().toISOString(),
            },
          ],
        }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    await expect(page.locator('.document-badge')).toBeVisible();
    await expect(page.locator('.document-badge')).toContainText('2 documents loaded');
  });

  test('should persist theme preference in localStorage', async ({ page, context }) => {
    // Set initial theme in localStorage
    await context.addCookies([]);
    await page.goto('/');
    
    const themeToggle = page.locator('.theme-toggle');
    await themeToggle.click();
    await page.waitForTimeout(500);

    // Reload page
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Check theme is persisted
    const html = page.locator('html');
    await expect(html).toHaveAttribute('data-theme', 'dark');
  });
});
