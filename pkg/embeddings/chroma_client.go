package embeddings

import (
	"context"
	"fmt"
	"log"

	chroma "github.com/amikos-tech/chroma-go"
	"github.com/amikos-tech/chroma-go/types"
)

type ChromaClient struct {
	ch  *chroma.Client
	emb Embedder
}

var _ Client = (*ChromaClient)(nil)

func NewChromaClient(addr string, emb Embedder) (*ChromaClient, error) {
	ch, err := chroma.NewClient(addr)
	if err != nil {
		return nil, err
	}
	return &ChromaClient{
		ch:  ch,
		emb: emb,
	}, nil
}

func (c *ChromaClient) CreateSchema(ctx context.Context, model Model, cfg *SchemaConfig) error {
	collName := c.getCollectionNameForModel(model)
	_, err := c.ch.CreateCollection(
		ctx,
		collName,
		map[string]any{},
		/*createOrGet=*/ false,
		newChromaEmbedder(c.emb),
		// TODO(osdrv): collect distance function from cfg
		types.L2,
	)
	if err != nil {
		return fmt.Errorf("Failed to create ChromaDB collection: %s", err)
	}
	return nil
}

func (c *ChromaClient) DropSchema(ctx context.Context, model Model) error {
	_, err := c.ch.DeleteCollection(ctx, c.getCollectionNameForModel(model))
	return err
}

func (c *ChromaClient) InsertDocument(ctx context.Context, model Model, doc *Document) error {
	return c.InsertAllDocuments(ctx, model, []*Document{doc})
}

func (c *ChromaClient) InsertAllDocuments(ctx context.Context, model Model, docs []*Document) error {
	coll, err := c.getCollection(ctx, c.getCollectionNameForModel(model), c.emb)
	if err != nil {
		return fmt.Errorf("Failed to get Chroma collection: %s", err)
	}
	embs := make([]*types.Embedding, 0, len(docs))
	metas := make([]map[string]any, 0, len(docs))
	texts := make([]string, 0, len(docs))
	ids := make([]string, 0, len(docs))

	for _, doc := range docs {
		if doc.Embedding == nil {
			v64, err := c.emb.Embed(ctx, doc.Text)
			if err != nil {
				return fmt.Errorf("Failed to embed document: %s", err)
			}
			doc.Embedding = v64
		}
		v32 := f32(doc.Embedding)
		embs = append(embs, &types.Embedding{
			ArrayOfFloat32: &v32,
		})
		metas = append(metas, map[string]any{})
		texts = append(texts, doc.Text)
		ids = append(ids, doc.FileId)
	}
	if _, err := coll.Add(ctx, embs, metas, texts, ids); err != nil {
		return fmt.Errorf("Failed to add documents: %s", err)
	}
	return nil
}

func (c *ChromaClient) DeleteDocument(ctx context.Context, model Model, docKey string) error {
	col, err := c.getCollection(ctx, c.getCollectionNameForModel(model), c.emb)
	if err != nil {
		return fmt.Errorf("Failed to get Chroma collection: %s", err)
	}
	if _, err := col.Delete(ctx, []string{docKey}, nil, nil); err != nil {
		return err
	}
	return nil
}

func (c *ChromaClient) DeleteAllDocuments(ctx context.Context, model Model) error {
	col, err := c.getCollection(ctx, c.getCollectionNameForModel(model), c.emb)
	if err != nil {
		return fmt.Errorf("Failed to get Chroma collection: %s", err)
	}
	if _, err := col.Delete(ctx, nil, nil, nil); err != nil {
		return err
	}
	return nil
}

func (c *ChromaClient) FindKNearest(ctx context.Context, model Model, doc *Document, k int) ([]*Document, error) {
	col, err := c.getCollection(ctx, c.getCollectionNameForModel(model), c.emb)
	if err != nil {
		return nil, fmt.Errorf("Failed to get Chroma collection: %s", err)
	}
	res, err := col.Query(ctx, []string{doc.Text}, int32(k), nil, nil, []types.QueryEnum{types.IDocuments, types.IDistances})
	if err != nil {
		return nil, err
	}

	log.Printf("Result: %+v", res)

	docs := make([]*Document, 0, len(res.Documents))
	for i := 0; i < len(res.Documents); i++ {
		for j := 0; j < len(res.Documents[i]); j++ {
			docs = append(docs, &Document{
				Text:   res.Documents[i][j],
				FileId: res.Ids[i][j],
			})
		}
	}

	return docs, nil
}

func f32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

func (c *ChromaClient) getCollection(ctx context.Context, collName string, emb Embedder) (*chroma.Collection, error) {
	return c.ch.GetCollection(ctx, collName, newChromaEmbedder(emb))
}

func (c *ChromaClient) getCollectionNameForModel(model Model) string {
	return fmt.Sprintf("embeddings-%s", model.Name())
}

type chromaEmbedder struct {
	emb Embedder
}

var _ types.EmbeddingFunction = (*chromaEmbedder)(nil)

func newChromaEmbedder(emb Embedder) types.EmbeddingFunction {
	return &chromaEmbedder{
		emb: emb,
	}
}

func (e *chromaEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([]*types.Embedding, error) {
	embs := make([]*types.Embedding, 0, len(texts))
	for _, text := range texts {
		emb, err := e.EmbedQuery(ctx, text)
		if err != nil {
			return nil, err
		}
		embs = append(embs, emb)
	}
	return embs, nil
}

func (e *chromaEmbedder) EmbedQuery(ctx context.Context, text string) (*types.Embedding, error) {
	vector, err := e.emb.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	v32 := f32(vector)
	return &types.Embedding{
		ArrayOfFloat32: &v32,
	}, nil
}

func (e *chromaEmbedder) EmbedRecords(ctx context.Context, records []*types.Record, force bool) error {
	panic("Not implemented")
}
