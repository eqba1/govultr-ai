# Vultr Inference Go SDK

A comprehensive Go SDK for the [Vultr Inference API](https://api.vultrinference.com/), providing easy-to-use interfaces for chat completions, image generation, text-to-speech, vector store operations, and more.

## Features

- 🤖 **Chat Completions**: Support for text generation models with streaming capabilities
- 🔍 **RAG (Retrieval-Augmented Generation)**: Enhanced chat completions with vector store context
- 🖼️ **Image Generation**: Create images from text prompts
- 🔊 **Text-to-Speech**: Convert text to audio
- 📚 **Vector Store Operations**: Manage collections, items, and files for semantic search
- 📊 **Usage Tracking**: Monitor API usage and costs
- 📝 **Request Logging**: Access detailed request/response logs
- ⚡ **Streaming Support**: Real-time response streaming for chat completions
- 🛡️ **Type Safety**: Full type definitions for all API endpoints
- ✅ **Comprehensive Tests**: Extensive test coverage with examples

## Installation

```bash
go get github.com/eqba1/go-vultrai
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    vultrai "github.com/eqba1/go-vultrai"
)

func main() {
    client := vultrai.NewClient("your-api-key")
    
    response, err := client.SimpleChatCompletion(
        context.Background(),
        vultrai.Llama33_70bInstructFp8,
        "Tell me about Go programming language",
    )
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Choices[0].Message.Content)
}
```

## Configuration

### Basic Client

```go
client := vultrai.NewClient("your-api-key")
```

### Custom Configuration

```go
import "time"

httpClient := &http.Client{
    Timeout: 60 * time.Second,
}

client := vultrai.NewClient(
    "your-api-key",
    vultrai.WithBaseURL("https://custom.api.endpoint.com/v1"),
    vultrai.WithHTTPClient(httpClient),
)
```

## Usage Examples

### Chat Completions

#### Simple Chat

```go
response, err := client.SimpleChatCompletion(
    ctx, 
    vultrai.Llama33_70bInstructFp8,
    "What is artificial intelligence?",
)
```

#### Advanced Chat with Options

```go
messages := []vultrai.Message{
    vultrai.CreateSystemMessage("You are a helpful coding assistant."),
    vultrai.CreateUserMessage("How do I implement a binary tree in Go?"),
}

response, err := client.ChatWithMessages(
    ctx,
    vultrai.Llama33_70bInstructFp8,
    messages,
    vultrai.WithMaxTokens(200),
    vultrai.WithTemperature(0.7),
    vultrai.WithTopP(0.9),
    vultrai.WithFrequencyPenalty(0.1),
)
```

#### Streaming Chat

```go
request := vultrai.ChatCompletionRequest{
    Model: vultrai.Llama33_70bInstructFp8,
    Messages: []vultrai.Message{
        {Role: "user", Content: "Write a short story"},
    },
}

err := client.StreamChatCompletion(ctx, request, func(chunk *vultrai.StreamChatCompletion) error {
    if len(chunk.Choices) > 0 {
        fmt.Print(chunk.Choices[0].Delta.Content)
    }
    return nil
})
```

### RAG (Retrieval-Augmented Generation)

```go
// Create a collection
collection, err := client.CreateCollection(ctx, vultrai.CreateCollectionRequest{
    Name: "knowledge-base",
})

// Add documents
client.AddItem(ctx, collection.Collection.ID, vultrai.AddItemRequest{
    Content: "Go is a programming language developed by Google.",
    Description: "Go language info",
    AutoChunk: vultrai.Bool(true),
})

// Query with context
response, err := client.CreateRAGChatCompletion(ctx, vultrai.RAGChatCompletionRequest{
    Collection: collection.Collection.ID,
    Model: vultrai.Llama33_70bInstructFp8,
    Messages: []vultrai.Message{
        {Role: "user", Content: "Who developed Go?"},
    },
})
```
