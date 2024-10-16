package embeddings

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"

	ollama "github.com/ollama/ollama/api"
)

type OllamaEmbedder struct {
	oc    *ollama.Client
	model Model
}

var _ Embedder = (*OllamaEmbedder)(nil)

func NewOllamaEmbedder(addr string, httpc *http.Client, model Model) (*OllamaEmbedder, error) {
	ctx := context.Background()
	url, err := url.Parse(addr)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse ollama address: %s", err)
	}
	oc := ollama.NewClient(url, httpc)
	if v, err := oc.Version(ctx); err != nil {
		return nil, fmt.Errorf("Failed to get Ollama version: %s", err)
	} else {
		log.Printf("Ollama version: %s", v)
	}

	return &OllamaEmbedder{
		oc:    oc,
		model: model,
	}, nil
}

func (e *OllamaEmbedder) Embed(ctx context.Context, prompt string) ([]float64, error) {
	res, err := e.oc.Embeddings(ctx, &ollama.EmbeddingRequest{
		Model:  e.model.Name(),
		Prompt: prompt,
	})
	if err != nil {
		return nil, err
	}
	return res.Embedding, nil
}
