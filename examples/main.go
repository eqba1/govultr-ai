package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	vultrai "github.com/eqba1/govultr-ai"
)

// DefaultModel is the default model to use in examples
var DefaultModel = vultrai.Qwen25_32bInstruct

func main() {
	// You can call any of the example functions here to see their output
	ExampleClient_SimpleChatCompletion()
	// *
}

func ExampleClient_SimpleChatCompletion() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	response, err := client.SimpleChatCompletion(
		context.Background(),
		DefaultModel,
		"فارسی من چطوره؟",
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response.Choices[0].Message.Content)
}

func ExampleClient_ChatWithMessages() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	messages := []vultrai.Message{
		vultrai.CreateSystemMessage("You are a helpful coding assistant."),
		vultrai.CreateUserMessage("How do I reverse a string in Go?"),
	}

	response, err := client.ChatWithMessages(
		context.Background(),
		DefaultModel,
		messages,
		vultrai.WithMaxTokens(150),
		vultrai.WithTemperature(0.7),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response.Choices[0].Message.Content)
}

func ExampleClient_CreateChatCompletion_advanced() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	request := vultrai.ChatCompletionRequest{
		Model: DefaultModel,
		Messages: []vultrai.Message{
			{Role: "system", Content: "You are a creative writer."},
			{Role: "user", Content: "Write a short story about a robot discovering emotions."},
		},
		MaxTokens:        vultrai.Int(500),
		Temperature:      vultrai.Float64(0.8),
		TopP:             vultrai.Float64(0.9),
		FrequencyPenalty: vultrai.Float64(0.1),
		PresencePenalty:  vultrai.Float64(0.1),
		Stop:             []string{"THE END"},
	}

	response, err := client.CreateChatCompletion(context.Background(), request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Story: %s\n", response.Choices[0].Message.Content)
	fmt.Printf("Tokens used: %d\n", response.Usage.TotalTokens)
}

func ExampleClient_StreamChatCompletion() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	request := vultrai.ChatCompletionRequest{
		Model: DefaultModel,
		Messages: []vultrai.Message{
			{Role: "user", Content: "Tell me about artificial intelligence"},
		},
	}

	fmt.Print("AI Response: ")
	err := client.StreamChatCompletion(context.Background(), request, func(chunk *vultrai.StreamChatCompletion) error {
		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

func ExampleClient_CreateRAGChatCompletion() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	// First, create a collection and add some documents
	collection, err := client.CreateCollection(context.Background(), vultrai.CreateCollectionRequest{
		Name: "company-docs",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Add a document to the collection
	_, err = client.AddItem(context.Background(), collection.Collection.ID, vultrai.AddItemRequest{
		Content:     "Our company was founded in 2020 and specializes in AI solutions.",
		Description: "Company history document",
		AutoChunk:   vultrai.Bool(true),
	})
	if err != nil {
		log.Fatal(err)
	}

	// Now query with RAG
	request := vultrai.RAGChatCompletionRequest{
		Collection: collection.Collection.ID,
		Model:      DefaultModel,
		Messages: []vultrai.Message{
			{Role: "user", Content: "When was the company founded?"},
		},
		MaxTokens: vultrai.Int(100),
	}

	response, err := client.CreateRAGChatCompletion(context.Background(), request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("RAG Response:", response.Choices[0].Message.Content)
}

func ExampleClient_GenerateImage() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	response, err := client.SimpleImageGeneration(
		context.Background(),
		"A futuristic city with flying cars at sunset",
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Generated %d image(s)\n", len(response.Data))
	for i, img := range response.Data {
		fmt.Printf("Image %d URL: %s\n", i+1, img.URL)
	}
}

func ExampleClient_GenerateImageWithOptions() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	response, err := client.GenerateImageWithOptions(
		context.Background(),
		"A majestic dragon in a medieval setting",
		vultrai.WithImageModel("flux.1-dev"),
		vultrai.WithImageCount(2),
		vultrai.WithImageSize("1024x1024"),
		vultrai.WithImageFormat("url"),
	)
	if err != nil {
		log.Fatal(err)
	}

	for i, img := range response.Data {
		fmt.Printf("Dragon image %d: %s\n", i+1, img.URL)
	}
}

func ExampleClient_CreateSpeech() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	request := vultrai.TTSRequest{
		Model: "tts-1",
		Input: "Hello! This is a test of the text-to-speech functionality.",
		Voice: "alloy",
	}

	audioData, err := client.CreateSpeech(context.Background(), request)
	if err != nil {
		log.Fatal(err)
	}

	// Save the audio to a file
	err = os.WriteFile("output.mp3", audioData, 0644)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Audio saved to output.mp3 (%d bytes)\n", len(audioData))
}

func ExampleClient_AddFile() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	// Create a collection
	collection, err := client.CreateCollection(context.Background(), vultrai.CreateCollectionRequest{
		Name: "document-collection",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Add a file (example with string reader)
	fileContent := "This is the content of my document file."
	fileReader := strings.NewReader(fileContent)

	fileResponse, err := client.AddFile(context.Background(), collection.Collection.ID, fileReader, "document.txt")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("File uploaded: %s (Status: %s)\n", fileResponse.File.Filename, fileResponse.File.Status)
}

func ExampleClient_GetUsage() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	usage, err := client.GetUsage(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Current Month Usage:\n")
	fmt.Printf("  Chat: $%.2f\n", usage.CurrentMonth.Chat)
	fmt.Printf("  TTS: $%.2f\n", usage.CurrentMonth.TTS)
	fmt.Printf("  Images: $%.2f\n", usage.CurrentMonth.Image)

	fmt.Printf("Previous Month Usage:\n")
	fmt.Printf("  Chat: $%.2f\n", usage.PreviousMonth.Chat)
	fmt.Printf("  TTS: $%.2f\n", usage.PreviousMonth.TTS)
	fmt.Printf("  Images: $%.2f\n", usage.PreviousMonth.Image)
}

func ExampleClient_GetRequestLogs() {
	client := vultrai.NewClient(os.Getenv("API_KEY"))

	request := vultrai.RequestLogsRequest{
		Period:   30, // Last 30 minutes
		Endpoint: "chat/completions",
	}

	logs, err := client.GetRequestLogs(context.Background(), request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Found %d requests:\n", len(logs.Requests))
	for _, req := range logs.Requests {
		fmt.Printf("- %s %s (Status: %d) at %s\n",
			req.Method, req.Endpoint, req.ResponseCode, req.Timestamp)
	}
}

// func ExampleValidation() {
// 	// Validate parameters before making requests
// 	temperature := 0.8
// 	if err := vultrai.ValidateTemperature(temperature); err != nil {
// 		log.Fatal(err)
// 	}

// 	topP := 0.9
// 	if err := vultrai.ValidateTopP(topP); err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Println("All parameters are valid!")
// }

// func ExampleStreamingWithAccumulation() {
// 	client := vultrai.NewClient(os.Getenv("API_KEY"))

// 	request := vultrai.ChatCompletionRequest{
// 		Model: DefaultModel,
// 		Messages: []vultrai.Message{
// 			{Role: "user", Content: "Explain quantum computing"},
// 		},
// 	}

// 	// Collect all chunks
// 	var chunks []*vultrai.StreamChatCompletion

// 	err := client.StreamChatCompletion(context.Background(), request, func(chunk *vultrai.StreamChatCompletion) error {
// 		chunks = append(chunks, chunk)
// 		// Print each chunk as it arrives
// 		if len(chunk.Choices) > 0 {
// 			fmt.Print(chunk.Choices[0].Delta.Content)
// 		}
// 		return nil
// 	})
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Println() // New line after streaming

// 	// Convert to complete response
// 	complete := vultrai.StreamToComplete(chunks)
// 	if complete != nil {
// 		fmt.Printf("\nFull response length: %d characters\n", len(complete.Choices[0].Message.Content))
// 	}

// 	// Or just accumulate content
// 	fullContent := vultrai.AccumulateStreamContent(chunks)
// 	fmt.Printf("Accumulated content: %s\n", fullContent)
// }

// func ExampleClient_VectorStore() {
// 	client := vultrai.NewClient(os.Getenv("API_KEY"))

// 	// Create a collection
// 	collection, err := client.CreateCollection(context.Background(), vultrai.CreateCollectionRequest{
// 		Name: "my-knowledge-base",
// 	})
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	fmt.Printf("Created collection: %s\n", collection.Collection.ID)

// 	// Add some items
// 	documents := []string{
// 		"The capital of France is Paris.",
// 		"Python is a programming language.",
// 		"Machine learning is a subset of artificial intelligence.",
// 	}

// 	for i, doc := range documents {
// 		item, err := client.AddItem(context.Background(), collection.Collection.ID, vultrai.AddItemRequest{
// 			Content:     doc,
// 			Description: fmt.Sprintf("Document %d", i+1),
// 		})
// 		if err != nil {
// 			log.Fatal(err)
// 		}
// 		fmt.Printf("Added item: %s\n", item.Item.ID)
// 	}

// 	// Search the collection
// 	searchResults, err := client.SearchCollection(context.Background(), collection.Collection.ID, vultrai.SearchRequest{
// 		Input: "What is the capital of France?",
// 	})
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Printf("Found %d results:\n", len(searchResults.Results))
// 	for _, result := range searchResults.Results {
// 		fmt.Printf("- %s\n", result.Content)
// 	}
// }
