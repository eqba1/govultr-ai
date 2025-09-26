package vultrai

import (
	"context"
	"fmt"
)

// Helper functions for common use cases

// SimpleChatCompletion is a helper function for simple chat completions
func (c *Client) SimpleChatCompletion(ctx context.Context, model, prompt string) (*ChatCompletionResponse, error) {
	req := ChatCompletionRequest{
		Model: model,
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	return c.CreateChatCompletion(ctx, req)
}

// ChatWithMessages is a helper function for chat completions with message history
func (c *Client) ChatWithMessages(ctx context.Context, model string, messages []Message, options ...ChatOption) (*ChatCompletionResponse, error) {
	req := ChatCompletionRequest{
		Model:    model,
		Messages: messages,
	}

	// Apply options
	for _, option := range options {
		option(&req)
	}

	return c.CreateChatCompletion(ctx, req)
}

// ChatOption represents a function to configure chat completion requests
type ChatOption func(*ChatCompletionRequest)

// WithMaxTokens sets the maximum number of tokens to generate
func WithMaxTokens(maxTokens int) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.MaxTokens = &maxTokens
	}
}

// WithTemperature sets the temperature for randomness
func WithTemperature(temperature float64) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.Temperature = &temperature
	}
}

// WithTopP sets the top-p value for nucleus sampling
func WithTopP(topP float64) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.TopP = &topP
	}
}

// WithSeed sets the seed for deterministic outputs
func WithSeed(seed int) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.Seed = &seed
	}
}

// WithStream enables/disables streaming
func WithStream(stream bool) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.Stream = &stream
	}
}

// WithFrequencyPenalty sets the frequency penalty
func WithFrequencyPenalty(penalty float64) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.FrequencyPenalty = &penalty
	}
}

// WithPresencePenalty sets the presence penalty
func WithPresencePenalty(penalty float64) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.PresencePenalty = &penalty
	}
}

// WithStopSequences sets the stop sequences
func WithStopSequences(stop []string) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.Stop = stop
	}
}

// WithLogProbs enables log probabilities
func WithLogProbs(logProbs bool, topLogProbs ...int) ChatOption {
	return func(req *ChatCompletionRequest) {
		req.LogProbs = &logProbs
		if len(topLogProbs) > 0 {
			req.TopLogProbs = &topLogProbs[0]
		}
	}
}

// SimpleImageGeneration is a helper function for simple image generation
func (c *Client) SimpleImageGeneration(ctx context.Context, prompt string) (*ImageGenerationResponse, error) {
	req := ImageGenerationRequest{
		Prompt: prompt,
	}

	return c.GenerateImage(ctx, req)
}

// GenerateImageWithOptions is a helper function for image generation with options
func (c *Client) GenerateImageWithOptions(ctx context.Context, prompt string, options ...ImageOption) (*ImageGenerationResponse, error) {
	req := ImageGenerationRequest{
		Prompt: prompt,
	}

	// Apply options
	for _, option := range options {
		option(&req)
	}

	return c.GenerateImage(ctx, req)
}

// ImageOption represents a function to configure image generation requests
type ImageOption func(*ImageGenerationRequest)

// WithImageModel sets the model for image generation
func WithImageModel(model string) ImageOption {
	return func(req *ImageGenerationRequest) {
		req.Model = model
	}
}

// WithImageCount sets the number of images to generate
func WithImageCount(n int) ImageOption {
	return func(req *ImageGenerationRequest) {
		req.N = &n
	}
}

// WithImageSize sets the size of generated images
func WithImageSize(size string) ImageOption {
	return func(req *ImageGenerationRequest) {
		req.Size = size
	}
}

// WithImageFormat sets the response format for images
func WithImageFormat(format string) ImageOption {
	return func(req *ImageGenerationRequest) {
		req.ResponseFormat = format
	}
}

// CreateSystemMessage creates a system message
func CreateSystemMessage(content string) Message {
	return Message{
		Role:    "system",
		Content: content,
	}
}

// CreateUserMessage creates a user message
func CreateUserMessage(content string) Message {
	return Message{
		Role:    "user",
		Content: content,
	}
}

// CreateAssistantMessage creates an assistant message
func CreateAssistantMessage(content string) Message {
	return Message{
		Role:    "assistant",
		Content: content,
	}
}

// Bool is a helper function to get a pointer to a bool value
func Bool(b bool) *bool {
	return &b
}

// Int is a helper function to get a pointer to an int value
func Int(i int) *int {
	return &i
}

// Float64 is a helper function to get a pointer to a float64 value
func Float64(f float64) *float64 {
	return &f
}

// ValidateTemperature validates temperature value
func ValidateTemperature(temperature float64) error {
	if temperature < 0.0 || temperature > 2.0 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0, got %f", temperature)
	}
	return nil
}

// ValidateTopP validates top-p value
func ValidateTopP(topP float64) error {
	if topP < 0.0 || topP > 1.0 {
		return fmt.Errorf("top_p must be between 0.0 and 1.0, got %f", topP)
	}
	return nil
}

// ValidateFrequencyPenalty validates frequency penalty value
func ValidateFrequencyPenalty(penalty float64) error {
	if penalty < -2.0 || penalty > 2.0 {
		return fmt.Errorf("frequency_penalty must be between -2.0 and 2.0, got %f", penalty)
	}
	return nil
}

// ValidatePresencePenalty validates presence penalty value
func ValidatePresencePenalty(penalty float64) error {
	if penalty < -2.0 || penalty > 2.0 {
		return fmt.Errorf("presence_penalty must be between -2.0 and 2.0, got %f", penalty)
	}
	return nil
}

// ValidateTopLogProbs validates top log probs value
func ValidateTopLogProbs(topLogProbs int) error {
	if topLogProbs < 0 || topLogProbs > 20 {
		return fmt.Errorf("top_logprobs must be between 0 and 20, got %d", topLogProbs)
	}
	return nil
}
