package vultrai

// Message represents a chat message in the conversation
type Message struct {
	Role      string     `json:"role"` // "system", "user", or "assistant"
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall represents a function call in the message
type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents the function details in a tool call
type Function struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionRequest represents the request for chat completion
type ChatCompletionRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	Stream           *bool     `json:"stream,omitempty"`
	MaxTokens        *int      `json:"max_tokens,omitempty"`
	N                *int      `json:"n,omitempty"`
	Seed             *int      `json:"seed,omitempty"`
	Temperature      *float64  `json:"temperature,omitempty"`
	TopP             *float64  `json:"top_p,omitempty"`
	FrequencyPenalty *float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64  `json:"presence_penalty,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	LogProbs         *bool     `json:"logprobs,omitempty"`
	TopLogProbs      *int      `json:"top_logprobs,omitempty"`
}

// RAGChatCompletionRequest represents the request for RAG chat completion
type RAGChatCompletionRequest struct {
	Collection       string    `json:"collection"`
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	MaxTokens        *int      `json:"max_tokens,omitempty"`
	N                *int      `json:"n,omitempty"`
	Seed             *int      `json:"seed,omitempty"`
	Temperature      *float64  `json:"temperature,omitempty"`
	TopP             *float64  `json:"top_p,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	FrequencyPenalty *float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64  `json:"presence_penalty,omitempty"`
	Stream           *bool     `json:"stream,omitempty"`
	LogProbs         *bool     `json:"logprobs,omitempty"`
	TopLogProbs      *int      `json:"top_logprobs,omitempty"`
}

// LogProb represents log probability information for a token
type LogProb struct {
	Token       string       `json:"token"`
	LogProb     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogProbs []TopLogProb `json:"top_logprobs,omitempty"`
}

// TopLogProb represents top log probability for a token
type TopLogProb struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

// LogProbs represents log probabilities for the content
type LogProbs struct {
	Content []LogProb `json:"content"`
}

// Choice represents a chat completion choice
type Choice struct {
	Index        int       `json:"index"`
	Message      Message   `json:"message"`
	LogProbs     *LogProbs `json:"logprobs,omitempty"`
	FinishReason string    `json:"finish_reason"`
}

// Usage represents token usage information
type Usage struct {
	CompletionTokens int `json:"completion_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse represents the response from chat completion
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// TTSRequest represents the request for text-to-speech
type TTSRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
	Voice string `json:"voice"`
}

// VectorStoreCollection represents a vector store collection
type VectorStoreCollection struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Created string `json:"created"`
}

// CreateCollectionRequest represents the request to create a collection
type CreateCollectionRequest struct {
	Name string `json:"name"`
}

// CreateCollectionResponse represents the response from creating a collection
type CreateCollectionResponse struct {
	Collection VectorStoreCollection `json:"collection"`
}

// UpdateCollectionRequest represents the request to update a collection
type UpdateCollectionRequest struct {
	Name string `json:"name"`
}

// UpdateCollectionResponse represents the response from updating a collection
type UpdateCollectionResponse struct {
	Collection VectorStoreCollection `json:"collection"`
}

// SearchRequest represents the request to search in a collection
type SearchRequest struct {
	Input string `json:"input"`
}

// SearchResult represents a search result
type SearchResult struct {
	ID      string `json:"id"`
	Created string `json:"created"`
	Content string `json:"content"`
}

// SearchResponse represents the response from search
type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Usage   Usage          `json:"usage"`
}

// CollectionItem represents an item in a vector store collection
type CollectionItem struct {
	ID          string `json:"id"`
	Created     string `json:"created"`
	Description string `json:"description"`
	Content     string `json:"content,omitempty"`
}

// ListItemsResponse represents the response from listing items
type ListItemsResponse struct {
	Items []CollectionItem `json:"items"`
}

// AddItemRequest represents the request to add an item to collection
type AddItemRequest struct {
	Content     string `json:"content"`
	Description string `json:"description,omitempty"`
	AutoChunk   *bool  `json:"auto_chunk,omitempty"`
}

// AddItemResponse represents the response from adding an item
type AddItemResponse struct {
	Item  CollectionItem `json:"item"`
	Usage Usage          `json:"usage"`
}

// GetItemResponse represents the response from getting an item
type GetItemResponse struct {
	Item CollectionItem `json:"item"`
}

// UpdateItemRequest represents the request to update an item
type UpdateItemRequest struct {
	Description string `json:"description"`
}

// UpdateItemResponse represents the response from updating an item
type UpdateItemResponse struct {
	Item CollectionItem `json:"item"`
}

// CollectionFile represents a file in a vector store collection
type CollectionFile struct {
	ID       string `json:"id"`
	Filename string `json:"filename"`
	Status   string `json:"status"` // "enqueued", "processing", "completed", "failed"
	Error    string `json:"error,omitempty"`
	Items    int    `json:"items"`
	Tokens   int    `json:"tokens"`
}

// ListFilesResponse represents the response from listing files
type ListFilesResponse struct {
	Files []CollectionFile `json:"files"`
}

// AddFileResponse represents the response from adding a file
type AddFileResponse struct {
	File CollectionFile `json:"file"`
}

// GetFileResponse represents the response from getting a file
type GetFileResponse struct {
	File CollectionFile `json:"file"`
}

// ImageGenerationRequest represents the request for image generation
type ImageGenerationRequest struct {
	Prompt         string `json:"prompt"`
	Model          string `json:"model,omitempty"`
	N              *int   `json:"n,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	Size           string `json:"size,omitempty"`
}

// ImageData represents generated image data
type ImageData struct {
	B64JSON string `json:"b64_json,omitempty"`
	URL     string `json:"url,omitempty"`
}

// ImageGenerationResponse represents the response from image generation
type ImageGenerationResponse struct {
	Created int64       `json:"created"`
	Data    []ImageData `json:"data"`
}

// MonthlyUsage represents usage for a month
type MonthlyUsage struct {
	Chat    float64 `json:"chat"`
	TTS     float64 `json:"tts"`
	TTSSM   float64 `json:"tts_sm"`
	Image   float64 `json:"image"`
	ImageSM float64 `json:"image_sm"`
}

// UsageResponse represents the response from usage endpoint
type UsageResponse struct {
	CurrentMonth  MonthlyUsage `json:"current_month"`
	PreviousMonth MonthlyUsage `json:"previous_month"`
}

// RequestLog represents a logged API request
type RequestLog struct {
	Timestamp      string `json:"timestamp"`
	Method         string `json:"method"`
	Endpoint       string `json:"endpoint"`
	RequestHeaders string `json:"request_headers"`
	RequestBody    string `json:"request_body"`
	ResponseBody   string `json:"response_body"`
	ResponseCode   int    `json:"response_code"`
}

// RequestLogsRequest represents the request for request logs
type RequestLogsRequest struct {
	Period    int    `json:"period"`              // 15, 30, 45, or 60 minutes
	Timestamp string `json:"timestamp,omitempty"` // UTC timestamp in ISO 8601 format
	Endpoint  string `json:"endpoint,omitempty"`  // Filter by endpoint name
}

// RequestLogsResponse represents the response from request logs
type RequestLogsResponse struct {
	Requests []RequestLog `json:"requests"`
}

// Error represents an API error response
type Error struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
}
