package vultrai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamReader(t *testing.T) {
	streamData := `data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"}}]}

data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]

`

	reader := NewStreamReader(io.NopCloser(strings.NewReader(streamData)))
	defer reader.Close()

	// First chunk
	chunk1, err := reader.Recv()
	require.NoError(t, err)
	assert.Equal(t, "chat-123", chunk1.ID)
	assert.Equal(t, "assistant", chunk1.Choices[0].Delta.Role)
	assert.Equal(t, "Hello", chunk1.Choices[0].Delta.Content)

	// Second chunk
	chunk2, err := reader.Recv()
	require.NoError(t, err)
	assert.Equal(t, " world", chunk2.Choices[0].Delta.Content)

	// Third chunk
	chunk3, err := reader.Recv()
	require.NoError(t, err)
	assert.Equal(t, "!", chunk3.Choices[0].Delta.Content)
	assert.Equal(t, "stop", *chunk3.Choices[0].FinishReason)

	// End of stream
	_, err = reader.Recv()
	assert.Equal(t, io.EOF, err)
}

func TestStreamReaderWithEmptyLines(t *testing.T) {
	streamData := `
data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"content":"test"}}]}


data: [DONE]
`

	reader := NewStreamReader(io.NopCloser(strings.NewReader(streamData)))
	defer reader.Close()

	chunk, err := reader.Recv()
	require.NoError(t, err)
	assert.Equal(t, "test", chunk.Choices[0].Delta.Content)

	_, err = reader.Recv()
	assert.Equal(t, io.EOF, err)
}

func TestStreamReaderInvalidJSON(t *testing.T) {
	streamData := `data: invalid json`

	reader := NewStreamReader(io.NopCloser(strings.NewReader(streamData)))
	defer reader.Close()

	_, err := reader.Recv()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "error parsing streaming response")
}

func TestAccumulateStreamContent(t *testing.T) {
	chunks := []*StreamChatCompletion{
		{
			ID: "chat-123",
			Choices: []StreamChoice{
				{Delta: StreamDelta{Content: "Hello"}},
			},
		},
		{
			ID: "chat-123",
			Choices: []StreamChoice{
				{Delta: StreamDelta{Content: " world"}},
			},
		},
		{
			ID: "chat-123",
			Choices: []StreamChoice{
				{Delta: StreamDelta{Content: "!"}},
			},
		},
	}

	content := AccumulateStreamContent(chunks)
	assert.Equal(t, "Hello world!", content)
}

func TestStreamToComplete(t *testing.T) {
	chunks := []*StreamChatCompletion{
		{
			ID:      "chat-123",
			Created: 1640995200,
			Model:   "test-model",
			Choices: []StreamChoice{
				{
					Index: 0,
					Delta: StreamDelta{
						Role:    "assistant",
						Content: "Hello",
					},
				},
			},
		},
		{
			ID:      "chat-123",
			Created: 1640995200,
			Model:   "test-model",
			Choices: []StreamChoice{
				{
					Index: 0,
					Delta: StreamDelta{
						Content: " world",
					},
				},
			},
		},
		{
			ID:      "chat-123",
			Created: 1640995200,
			Model:   "test-model",
			Choices: []StreamChoice{
				{
					Index: 0,
					Delta: StreamDelta{
						Content: "!",
					},
					FinishReason: stringPtr("stop"),
				},
			},
		},
	}

	complete := StreamToComplete(chunks)
	require.NotNil(t, complete)
	assert.Equal(t, "chat-123", complete.ID)
	assert.Equal(t, int64(1640995200), complete.Created)
	assert.Equal(t, "test-model", complete.Model)
	assert.Len(t, complete.Choices, 1)
	assert.Equal(t, "assistant", complete.Choices[0].Message.Role)
	assert.Equal(t, "Hello world!", complete.Choices[0].Message.Content)
	assert.Equal(t, "stop", complete.Choices[0].FinishReason)
}

func TestStreamToCompleteEmpty(t *testing.T) {
	complete := StreamToComplete([]*StreamChatCompletion{})
	assert.Nil(t, complete)
}

func TestCreateChatCompletionStream(t *testing.T) {
	client, mockTransport := setupTestClient()

	streamData := `data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}

data: [DONE]

`

	mockTransport.responses["POST /chat/completions"] = &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(streamData)),
	}

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}

	stream, err := client.CreateChatCompletionStream(context.Background(), req)
	require.NoError(t, err)
	defer stream.Close()

	chunk, err := stream.Recv()
	require.NoError(t, err)
	assert.Equal(t, "chat-123", chunk.ID)
	assert.Equal(t, "Hello", chunk.Choices[0].Delta.Content)

	_, err = stream.Recv()
	assert.Equal(t, io.EOF, err)

	// Verify streaming was enabled in request
	requests := mockTransport.GetRequests()
	require.Len(t, requests, 1)
	assert.Equal(t, "text/event-stream", requests[0].Header.Get("Accept"))
}

func TestStreamChatCompletion(t *testing.T) {
	client, mockTransport := setupTestClient()

	streamData := `data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"}}]}

data: [DONE]

`

	mockTransport.responses["POST /chat/completions"] = &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(streamData)),
	}

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}

	var receivedChunks []*StreamChatCompletion
	callback := func(chunk *StreamChatCompletion) error {
		receivedChunks = append(receivedChunks, chunk)
		return nil
	}

	err := client.StreamChatCompletion(context.Background(), req, callback)
	require.NoError(t, err)

	assert.Len(t, receivedChunks, 2)
	assert.Equal(t, "Hello", receivedChunks[0].Choices[0].Delta.Content)
	assert.Equal(t, " world", receivedChunks[1].Choices[0].Delta.Content)
}

func TestStreamChatCompletionCallbackError(t *testing.T) {
	client, mockTransport := setupTestClient()

	streamData := `data: {"id":"chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: [DONE]

`

	mockTransport.responses["POST /chat/completions"] = &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(streamData)),
	}

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}

	callback := func(chunk *StreamChatCompletion) error {
		return assert.AnError
	}

	err := client.StreamChatCompletion(context.Background(), req, callback)
	require.Error(t, err)
	assert.Equal(t, assert.AnError, err)
}

func TestCreateRAGChatCompletionStream(t *testing.T) {
	client, mockTransport := setupTestClient()

	streamData := `data: {"id":"rag-chat-123","created":1640995200,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Based on context"}}]}

data: [DONE]

`

	mockTransport.responses["POST /chat/completions/rag"] = &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(streamData)),
	}

	req := RAGChatCompletionRequest{
		Collection: "test-collection",
		Model:      "test-model",
		Messages:   []Message{{Role: "user", Content: "What does the doc say?"}},
	}

	stream, err := client.CreateRAGChatCompletionStream(context.Background(), req)
	require.NoError(t, err)
	defer stream.Close()

	chunk, err := stream.Recv()
	require.NoError(t, err)
	assert.Equal(t, "rag-chat-123", chunk.ID)
	assert.Equal(t, "Based on context", chunk.Choices[0].Delta.Content)
}

// Helper function for tests
func stringPtr(s string) *string {
	return &s
}
