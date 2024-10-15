package embeddings

import "context"

type Embedder interface {
	Embed(ctx context.Context, prompt string) ([]float64, error)
}
