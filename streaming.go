package vultrai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// StreamChatCompletion represents a streaming chat completion chunk
type StreamChatCompletion struct {
	ID      string         `json:"id"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// StreamChoice represents a streaming choice
type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	LogProbs     *LogProbs   `json:"logprobs,omitempty"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

// StreamDelta represents the delta in a streaming response
type StreamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// StreamReader wraps the streaming response reader
type StreamReader struct {
	reader  *bufio.Scanner
	closer  io.Closer
	isFirst bool
}

// NewStreamReader creates a new stream reader
func NewStreamReader(reader io.ReadCloser) *StreamReader {
	return &StreamReader{
		reader:  bufio.NewScanner(reader),
		closer:  reader,
		isFirst: true,
	}
}

// Recv receives the next streaming chunk
func (s *StreamReader) Recv() (*StreamChatCompletion, error) {
	for s.reader.Scan() {
		line := s.reader.Text()

		// Skip empty lines
		if line == "" {
			continue
		}

		// Check for data prefix
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		// Extract JSON data
		data := strings.TrimPrefix(line, "data: ")

		// Check for stream end
		if data == "[DONE]" {
			return nil, io.EOF
		}

		// Parse JSON
		var chunk StreamChatCompletion
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("error parsing streaming response: %w", err)
		}

		return &chunk, nil
	}

	if err := s.reader.Err(); err != nil {
		return nil, fmt.Errorf("error reading stream: %w", err)
	}

	return nil, io.EOF
}

// Close closes the stream reader
func (s *StreamReader) Close() error {
	if s.closer != nil {
		return s.closer.Close()
	}
	return nil
}

// CreateChatCompletionStream creates a streaming chat completion
func (c *Client) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*StreamReader, error) {
	// Ensure streaming is enabled
	req.Stream = Bool(true)

	resp, err := c.doRequest(ctx, "POST", "/chat/completions", req, map[string]string{
		"Accept": "text/event-stream",
	})
	if err != nil {
		return nil, err
	}

	return NewStreamReader(resp.Body), nil
}

// CreateRAGChatCompletionStream creates a streaming RAG chat completion
func (c *Client) CreateRAGChatCompletionStream(ctx context.Context, req RAGChatCompletionRequest) (*StreamReader, error) {
	// Ensure streaming is enabled
	req.Stream = Bool(true)

	resp, err := c.doRequest(ctx, "POST", "/chat/completions/rag", req, map[string]string{
		"Accept": "text/event-stream",
	})
	if err != nil {
		return nil, err
	}

	return NewStreamReader(resp.Body), nil
}

// StreamCallback represents a callback function for streaming responses
type StreamCallback func(*StreamChatCompletion) error

// StreamChatCompletion streams a chat completion with a callback
func (c *Client) StreamChatCompletion(ctx context.Context, req ChatCompletionRequest, callback StreamCallback) error {
	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return err
	}
	defer stream.Close()

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if err := callback(chunk); err != nil {
			return err
		}
	}

	return nil
}

// StreamRAGChatCompletion streams a RAG chat completion with a callback
func (c *Client) StreamRAGChatCompletion(ctx context.Context, req RAGChatCompletionRequest, callback StreamCallback) error {
	stream, err := c.CreateRAGChatCompletionStream(ctx, req)
	if err != nil {
		return err
	}
	defer stream.Close()

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if err := callback(chunk); err != nil {
			return err
		}
	}

	return nil
}

// AccumulateStreamContent accumulates content from streaming chunks
func AccumulateStreamContent(chunks []*StreamChatCompletion) string {
	var content strings.Builder

	for _, chunk := range chunks {
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	return content.String()
}

// StreamToComplete converts a streaming response to a complete response
func StreamToComplete(chunks []*StreamChatCompletion) *ChatCompletionResponse {
	if len(chunks) == 0 {
		return nil
	}

	// Use the first chunk as base
	first := chunks[0]

	// Accumulate content
	var content strings.Builder
	var finishReason string

	for _, chunk := range chunks {
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			content.WriteString(choice.Delta.Content)

			if choice.FinishReason != nil {
				finishReason = *choice.FinishReason
			}
		}
	}

	return &ChatCompletionResponse{
		ID:      first.ID,
		Created: first.Created,
		Model:   first.Model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: content.String(),
				},
				FinishReason: finishReason,
			},
		},
	}
}
