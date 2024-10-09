package main

import (
	"context"
	"log"
	"net/http"
	"net/url"

	ollama "github.com/ollama/ollama/api"
)

func main() {
	ctx := context.Background()
	httpc := &http.Client{}
	url, err := url.Parse("http://127.0.0.1:11434")
	if err != nil {
		log.Fatalf("Failed to parse ollama address: %s", err)
	}
	ollamaClient := ollama.NewClient(url, httpc)
	if v, err := ollamaClient.Version(ctx); err != nil {
		log.Fatalf("Failed to get Ollama version: %s", err)
	} else {
		log.Printf("Ollama version: %s", v)
	}
}
