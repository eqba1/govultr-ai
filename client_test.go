package vultrai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockTransport implements http.RoundTripper for testing
type MockTransport struct {
	responses map[string]*http.Response
	requests  []*http.Request
}

func NewMockTransport() *MockTransport {
	return &MockTransport{
		responses: make(map[string]*http.Response),
		requests:  make([]*http.Request, 0),
	}
}

func (m *MockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	m.requests = append(m.requests, req)

	key := req.Method + " " + req.URL.Path
	if resp, exists := m.responses[key]; exists {
		return resp, nil
	}

	// Default response
	return &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader("{}")),
	}, nil
}

func (m *MockTransport) SetResponse(method, path string, statusCode int, body interface{}) {
	var bodyReader io.Reader

	if body != nil {
		jsonBody, _ := json.Marshal(body)
		bodyReader = strings.NewReader(string(jsonBody))
	} else {
		bodyReader = strings.NewReader("")
	}

	m.responses[method+" "+path] = &http.Response{
		StatusCode: statusCode,
		Header:     make(http.Header),
		Body:       io.NopCloser(bodyReader),
	}
}

func (m *MockTransport) GetRequests() []*http.Request {
	return m.requests
}

func setupTestClient() (*Client, *MockTransport) {
	mockTransport := NewMockTransport()
	httpClient := &http.Client{Transport: mockTransport}

	client := NewClient("test-api-key", WithHTTPClient(httpClient))
	return client, mockTransport
}

func TestNewClient(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		options []ClientOption
		wantKey string
		wantURL string
	}{
		{
			name:    "basic client",
			apiKey:  "test-key",
			wantKey: "test-key",
			wantURL: defaultBaseURL,
		},
		{
			name:   "client with custom base URL",
			apiKey: "test-key",
			options: []ClientOption{
				WithBaseURL("https://custom.api.com/v1/"),
			},
			wantKey: "test-key",
			wantURL: "https://custom.api.com/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.apiKey, tt.options...)

			assert.Equal(t, tt.wantKey, client.apiKey)
			assert.Equal(t, tt.wantURL, client.baseURL)
			assert.NotNil(t, client.httpClient)
		})
	}
}

func TestCreateChatCompletion(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &ChatCompletionResponse{
		ID:      "chat-123",
		Created: 1640995200,
		Model:   "test-model",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "Hello! How can I help you?",
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			CompletionTokens: 10,
			PromptTokens:     5,
			TotalTokens:      15,
		},
	}

	mockTransport.SetResponse("POST", "/chat/completions", 200, expectedResp)

	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := client.CreateChatCompletion(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, expectedResp.ID, resp.ID)
	assert.Equal(t, expectedResp.Model, resp.Model)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, "Hello! How can I help you?", resp.Choices[0].Message.Content)

	// Verify request was made correctly
	requests := mockTransport.GetRequests()
	require.Len(t, requests, 1)
	assert.Equal(t, "POST", requests[0].Method)
	assert.Equal(t, "/chat/completions", requests[0].URL.Path)
	assert.Equal(t, "Bearer test-api-key", requests[0].Header.Get("Authorization"))
}

func TestCreateRAGChatCompletion(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &ChatCompletionResponse{
		ID:      "rag-chat-123",
		Created: 1640995200,
		Model:   "test-model",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "Based on the documents, here's the answer...",
				},
				FinishReason: "stop",
			},
		},
	}

	mockTransport.SetResponse("POST", "/chat/completions/rag", 200, expectedResp)

	req := RAGChatCompletionRequest{
		Collection: "test-collection",
		Model:      "test-model",
		Messages: []Message{
			{Role: "user", Content: "What does the document say?"},
		},
	}

	resp, err := client.CreateRAGChatCompletion(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, expectedResp.ID, resp.ID)
	assert.Equal(t, "Based on the documents, here's the answer...", resp.Choices[0].Message.Content)
}

func TestCreateSpeech(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedAudio := []byte("fake-audio-data")
	mockTransport.SetResponse("POST", "/audio/speech", 200, nil)

	// Override the response with raw audio data
	mockTransport.responses["POST /audio/speech"] = &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(bytes.NewReader(expectedAudio)),
	}

	req := TTSRequest{
		Model: "tts-model",
		Input: "Hello world",
		Voice: "female",
	}

	audio, err := client.CreateSpeech(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, expectedAudio, audio)
}

func TestCreateCollection(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &CreateCollectionResponse{
		Collection: VectorStoreCollection{
			ID:      "coll-123",
			Name:    "test-collection",
			Created: "2024-01-01T00:00:00Z",
		},
	}

	mockTransport.SetResponse("POST", "/vector-stores/collections", 200, expectedResp)

	req := CreateCollectionRequest{
		Name: "test-collection",
	}

	resp, err := client.CreateCollection(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "coll-123", resp.Collection.ID)
	assert.Equal(t, "test-collection", resp.Collection.Name)
}

func TestSearchCollection(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &SearchResponse{
		Results: []SearchResult{
			{
				ID:      "result-1",
				Created: "2024-01-01T00:00:00Z",
				Content: "This is relevant content",
			},
		},
		Usage: Usage{
			PromptTokens: 5,
			TotalTokens:  5,
		},
	}

	mockTransport.SetResponse("POST", "/vector-stores/collections/coll-123/search", 200, expectedResp)

	req := SearchRequest{
		Input: "search query",
	}

	resp, err := client.SearchCollection(context.Background(), "coll-123", req)
	require.NoError(t, err)
	assert.Len(t, resp.Results, 1)
	assert.Equal(t, "This is relevant content", resp.Results[0].Content)
}

func TestAddItem(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &AddItemResponse{
		Item: CollectionItem{
			ID:          "item-123",
			Created:     "2024-01-01T00:00:00Z",
			Description: "Test item",
			Content:     "This is test content",
		},
		Usage: Usage{
			PromptTokens: 10,
			TotalTokens:  10,
		},
	}

	mockTransport.SetResponse("POST", "/vector-stores/collections/coll-123/items", 200, expectedResp)

	req := AddItemRequest{
		Content:     "This is test content",
		Description: "Test item",
		AutoChunk:   Bool(true),
	}

	resp, err := client.AddItem(context.Background(), "coll-123", req)
	require.NoError(t, err)
	assert.Equal(t, "item-123", resp.Item.ID)
	assert.Equal(t, "This is test content", resp.Item.Content)
}

func TestGenerateImage(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &ImageGenerationResponse{
		Created: 1640995200,
		Data: []ImageData{
			{
				URL: "https://example.com/image1.png",
			},
			{
				URL: "https://example.com/image2.png",
			},
		},
	}

	mockTransport.SetResponse("POST", "/images/generations", 200, expectedResp)

	req := ImageGenerationRequest{
		Prompt: "A beautiful sunset",
		N:      Int(2),
		Size:   "1024x1024",
	}

	resp, err := client.GenerateImage(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, resp.Data, 2)
	assert.Equal(t, "https://example.com/image1.png", resp.Data[0].URL)
}

func TestGetUsage(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &UsageResponse{
		CurrentMonth: MonthlyUsage{
			Chat:  100.5,
			TTS:   50.2,
			Image: 25.1,
		},
		PreviousMonth: MonthlyUsage{
			Chat:  75.3,
			TTS:   30.1,
			Image: 15.0,
		},
	}

	mockTransport.SetResponse("GET", "/usage", 200, expectedResp)

	resp, err := client.GetUsage(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 100.5, resp.CurrentMonth.Chat)
	assert.Equal(t, 75.3, resp.PreviousMonth.Chat)
}

func TestErrorHandling(t *testing.T) {
	client, mockTransport := setupTestClient()

	errorResp := Error{
		Message: "Invalid request",
		Type:    "invalid_request_error",
		Code:    "invalid_api_key",
	}

	mockTransport.SetResponse("POST", "/chat/completions", 400, errorResp)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	_, err := client.CreateChatCompletion(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Invalid request")
}

func TestHelperFunctions(t *testing.T) {
	t.Run("CreateMessages", func(t *testing.T) {
		system := CreateSystemMessage("You are a helpful assistant")
		user := CreateUserMessage("Hello")
		assistant := CreateAssistantMessage("Hi there!")

		assert.Equal(t, "system", system.Role)
		assert.Equal(t, "You are a helpful assistant", system.Content)
		assert.Equal(t, "user", user.Role)
		assert.Equal(t, "Hello", user.Content)
		assert.Equal(t, "assistant", assistant.Role)
		assert.Equal(t, "Hi there!", assistant.Content)
	})

	t.Run("PointerHelpers", func(t *testing.T) {
		assert.Equal(t, true, *Bool(true))
		assert.Equal(t, 42, *Int(42))
		assert.Equal(t, 3.14, *Float64(3.14))
	})

	t.Run("Validators", func(t *testing.T) {
		assert.NoError(t, ValidateTemperature(0.5))
		assert.Error(t, ValidateTemperature(-1.0))
		assert.Error(t, ValidateTemperature(3.0))

		assert.NoError(t, ValidateTopP(0.9))
		assert.Error(t, ValidateTopP(-0.1))
		assert.Error(t, ValidateTopP(1.1))

		assert.NoError(t, ValidateFrequencyPenalty(1.0))
		assert.Error(t, ValidateFrequencyPenalty(-3.0))
		assert.Error(t, ValidateFrequencyPenalty(3.0))
	})
}

func TestChatOptions(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &ChatCompletionResponse{
		ID:      "chat-123",
		Created: 1640995200,
		Model:   "test-model",
		Choices: []Choice{
			{
				Index:        0,
				Message:      Message{Role: "assistant", Content: "Response"},
				FinishReason: "stop",
			},
		},
	}

	mockTransport.SetResponse("POST", "/chat/completions", 200, expectedResp)

	messages := []Message{
		CreateUserMessage("Hello"),
	}

	_, err := client.ChatWithMessages(context.Background(), "test-model", messages,
		WithMaxTokens(100),
		WithTemperature(0.7),
		WithTopP(0.9),
		WithFrequencyPenalty(0.1),
		WithPresencePenalty(0.2),
		WithStopSequences([]string{"END"}),
	)

	require.NoError(t, err)

	// Verify the request was made with correct parameters
	requests := mockTransport.GetRequests()
	require.Len(t, requests, 1)

	var reqBody ChatCompletionRequest
	err = json.NewDecoder(requests[0].Body).Decode(&reqBody)
	require.NoError(t, err)

	assert.Equal(t, 100, *reqBody.MaxTokens)
	assert.Equal(t, 0.7, *reqBody.Temperature)
	assert.Equal(t, 0.9, *reqBody.TopP)
	assert.Equal(t, 0.1, *reqBody.FrequencyPenalty)
	assert.Equal(t, 0.2, *reqBody.PresencePenalty)
	assert.Equal(t, []string{"END"}, reqBody.Stop)
}

func TestImageOptions(t *testing.T) {
	client, mockTransport := setupTestClient()

	expectedResp := &ImageGenerationResponse{
		Created: 1640995200,
		Data:    []ImageData{{URL: "https://example.com/image.png"}},
	}

	mockTransport.SetResponse("POST", "/images/generations", 200, expectedResp)

	_, err := client.GenerateImageWithOptions(context.Background(), "A cat",
		WithImageModel("flux.1-dev"),
		WithImageCount(1),
		WithImageSize("512x512"),
		WithImageFormat("url"),
	)

	require.NoError(t, err)

	// Verify the request parameters
	requests := mockTransport.GetRequests()
	require.Len(t, requests, 1)

	var reqBody ImageGenerationRequest
	err = json.NewDecoder(requests[0].Body).Decode(&reqBody)
	require.NoError(t, err)

	assert.Equal(t, "A cat", reqBody.Prompt)
	assert.Equal(t, "flux.1-dev", reqBody.Model)
	assert.Equal(t, 1, *reqBody.N)
	assert.Equal(t, "512x512", reqBody.Size)
	assert.Equal(t, "url", reqBody.ResponseFormat)
}
