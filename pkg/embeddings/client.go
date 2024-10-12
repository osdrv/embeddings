package embeddings

import "context"

type Client interface {
	CreateSchema(ctx context.Context, model Model, cfg *SchemaConfig) error
	DropSchema(ctx context.Context, model Model) error
	InsertDocument(ctx context.Context, model Model, doc *Document) error
	DeleteDocument(ctx context.Context, model Model, docKey string) error
	DeleteAllDocuments(ctx context.Context, model Model) error
	FindKNearest(ctx context.Context, model Model, doc *Document, k int) ([]*Document, error)
}
