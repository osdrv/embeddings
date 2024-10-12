package embeddings

import "context"

type Embedder interface {
	Embed(ctx context.Context, model Model, prompt string) ([]float64, error)
}
