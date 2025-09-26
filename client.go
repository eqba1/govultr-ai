package vultrai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	defaultBaseURL = "https://api.vultrinference.com/v1"
	defaultTimeout = 30 * time.Second

	// Content types
	contentTypeJSON = "application/json"
)

// Client represents the Vultr Inference API client
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// ClientOption represents a function to configure the client
type ClientOption func(*Client)

// WithBaseURL sets a custom base URL for the client
func WithBaseURL(baseURL string) ClientOption {
	return func(c *Client) {
		c.baseURL = strings.TrimSuffix(baseURL, "/")
	}
}

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

// NewClient creates a new Vultr Inference API client
func NewClient(apiKey string, options ...ClientOption) *Client {
	client := &Client{
		baseURL: defaultBaseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: defaultTimeout,
		},
	}

	for _, option := range options {
		option(client)
	}

	return client
}

// doRequest performs an HTTP request with proper error handling
func (c *Client) doRequest(ctx context.Context, method, endpoint string, body interface{}, headers map[string]string) (*http.Response, error) {
	var reqBody io.Reader

	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("error marshaling request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+endpoint, reqBody)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	// Set default headers
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", contentTypeJSON)
	req.Header.Set("Accept", contentTypeJSON)

	// Set custom headers
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}

	// Check for HTTP errors
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)

		var apiError Error
		if err := json.Unmarshal(body, &apiError); err != nil {
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
		}
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, apiError.Message)
	}

	return resp, nil
}

// doMultipartRequest performs a multipart form request
func (c *Client) doMultipartRequest(ctx context.Context, endpoint string, fields map[string]string, file io.Reader, filename string) (*http.Response, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add form fields
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			return nil, fmt.Errorf("error writing field %s: %w", key, err)
		}
	}

	// Add file if provided
	if file != nil && filename != "" {
		part, err := writer.CreateFormFile("file", filename)
		if err != nil {
			return nil, fmt.Errorf("error creating form file: %w", err)
		}

		if _, err := io.Copy(part, file); err != nil {
			return nil, fmt.Errorf("error copying file content: %w", err)
		}
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("error closing multipart writer: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+endpoint, &buf)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)

		var apiError Error
		if err := json.Unmarshal(body, &apiError); err != nil {
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
		}
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, apiError.Message)
	}

	return resp, nil
}

// CreateChatCompletion creates a chat completion
func (c *Client) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	resp, err := c.doRequest(ctx, "POST", "/chat/completions", req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &chatResp, nil
}

// CreateRAGChatCompletion creates a RAG chat completion
func (c *Client) CreateRAGChatCompletion(ctx context.Context, req RAGChatCompletionRequest) (*ChatCompletionResponse, error) {
	resp, err := c.doRequest(ctx, "POST", "/chat/completions/rag", req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &chatResp, nil
}

// CreateSpeech generates speech from text
func (c *Client) CreateSpeech(ctx context.Context, req TTSRequest) ([]byte, error) {
	resp, err := c.doRequest(ctx, "POST", "/audio/speech", req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading audio response: %w", err)
	}

	return audio, nil
}

// CreateCollection creates a new vector store collection
func (c *Client) CreateCollection(ctx context.Context, req CreateCollectionRequest) (*CreateCollectionResponse, error) {
	resp, err := c.doRequest(ctx, "POST", "/vector-stores/collections", req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var collResp CreateCollectionResponse
	if err := json.NewDecoder(resp.Body).Decode(&collResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &collResp, nil
}

// UpdateCollection updates a vector store collection
func (c *Client) UpdateCollection(ctx context.Context, id string, req UpdateCollectionRequest) (*UpdateCollectionResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s", id)
	resp, err := c.doRequest(ctx, "PUT", endpoint, req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var collResp UpdateCollectionResponse
	if err := json.NewDecoder(resp.Body).Decode(&collResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &collResp, nil
}

// SearchCollection searches items in a vector store collection
func (c *Client) SearchCollection(ctx context.Context, id string, req SearchRequest) (*SearchResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/search", id)
	resp, err := c.doRequest(ctx, "POST", endpoint, req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var searchResp SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &searchResp, nil
}

// ListItems lists items in a vector store collection
func (c *Client) ListItems(ctx context.Context, collectionID string) (*ListItemsResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/items", collectionID)
	resp, err := c.doRequest(ctx, "GET", endpoint, nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var itemsResp ListItemsResponse
	if err := json.NewDecoder(resp.Body).Decode(&itemsResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &itemsResp, nil
}

// AddItem adds an item to a vector store collection
func (c *Client) AddItem(ctx context.Context, collectionID string, req AddItemRequest) (*AddItemResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/items", collectionID)
	resp, err := c.doRequest(ctx, "POST", endpoint, req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var itemResp AddItemResponse
	if err := json.NewDecoder(resp.Body).Decode(&itemResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &itemResp, nil
}

// GetItem retrieves an item from a vector store collection
func (c *Client) GetItem(ctx context.Context, collectionID, itemID string) (*GetItemResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/items/%s", collectionID, itemID)
	resp, err := c.doRequest(ctx, "GET", endpoint, nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var itemResp GetItemResponse
	if err := json.NewDecoder(resp.Body).Decode(&itemResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &itemResp, nil
}

// UpdateItem updates an item in a vector store collection
func (c *Client) UpdateItem(ctx context.Context, collectionID, itemID string, req UpdateItemRequest) (*UpdateItemResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/items/%s", collectionID, itemID)
	resp, err := c.doRequest(ctx, "PUT", endpoint, req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var itemResp UpdateItemResponse
	if err := json.NewDecoder(resp.Body).Decode(&itemResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &itemResp, nil
}

// ListFiles lists files in a vector store collection
func (c *Client) ListFiles(ctx context.Context, collectionID string) (*ListFilesResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/files", collectionID)
	resp, err := c.doRequest(ctx, "GET", endpoint, nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var filesResp ListFilesResponse
	if err := json.NewDecoder(resp.Body).Decode(&filesResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &filesResp, nil
}

// AddFile adds a file to a vector store collection
func (c *Client) AddFile(ctx context.Context, collectionID string, file io.Reader, filename string) (*AddFileResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/files", collectionID)
	resp, err := c.doMultipartRequest(ctx, endpoint, nil, file, filename)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var fileResp AddFileResponse
	if err := json.NewDecoder(resp.Body).Decode(&fileResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &fileResp, nil
}

// GetFile retrieves a file from a vector store collection
func (c *Client) GetFile(ctx context.Context, collectionID, fileID string) (*GetFileResponse, error) {
	endpoint := fmt.Sprintf("/vector-stores/collections/%s/files/%s", collectionID, fileID)
	resp, err := c.doRequest(ctx, "GET", endpoint, nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var fileResp GetFileResponse
	if err := json.NewDecoder(resp.Body).Decode(&fileResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &fileResp, nil
}

// GenerateImage generates an image from a text prompt
func (c *Client) GenerateImage(ctx context.Context, req ImageGenerationRequest) (*ImageGenerationResponse, error) {
	resp, err := c.doRequest(ctx, "POST", "/images/generations", req, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var imgResp ImageGenerationResponse
	if err := json.NewDecoder(resp.Body).Decode(&imgResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &imgResp, nil
}

// GetUsage retrieves usage information
func (c *Client) GetUsage(ctx context.Context) (*UsageResponse, error) {
	resp, err := c.doRequest(ctx, "GET", "/usage", nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var usageResp UsageResponse
	if err := json.NewDecoder(resp.Body).Decode(&usageResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &usageResp, nil
}

// GetRequestLogs retrieves API request logs
func (c *Client) GetRequestLogs(ctx context.Context, req RequestLogsRequest) (*RequestLogsResponse, error) {
	// Build query parameters
	params := url.Values{}
	params.Set("period", strconv.Itoa(req.Period))
	if req.Timestamp != "" {
		params.Set("timestamp", req.Timestamp)
	}
	if req.Endpoint != "" {
		params.Set("endpoint", req.Endpoint)
	}

	endpoint := "/request-logs?" + params.Encode()
	resp, err := c.doRequest(ctx, "GET", endpoint, nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var logsResp RequestLogsResponse
	if err := json.NewDecoder(resp.Body).Decode(&logsResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &logsResp, nil
}
