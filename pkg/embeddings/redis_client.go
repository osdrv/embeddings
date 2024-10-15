package embeddings

import (
	"context"
	"fmt"
	"log"

	"github.com/redis/go-redis/v9"
)

type RedisClient struct {
	rc  *redis.Client
	emb Embedder
}

var _ Client = (*RedisClient)(nil)

func NewRedisClient(addr string, emb Embedder) (*RedisClient, error) {
	rc := redis.NewClient(&redis.Options{
		Addr: addr,
	})
	c := &RedisClient{
		rc:  rc,
		emb: emb,
	}

	if err := rc.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("Failed to ping redis: %s", err)
	}

	return c, nil
}

func (c *RedisClient) CreateSchema(ctx context.Context, model Model, cfg *SchemaConfig) error {
	index := c.getIndexForModel(model)
	indexInfo, err := c.getIndexInfo(ctx, index)
	if err != nil {
		return fmt.Errorf("Failed to fetch redis index info: %s", err)
	}
	if len(indexInfo) != 0 {
		return fmt.Errorf("Redis index %s already exists", index)
	}

	if err = c.rc.Do(ctx, "FT.CREATE", index,
		"PREFIX", "1", c.getPrefixForModel(model), "SCORE", "1.0",
		"SCHEMA",
		"file_id", "TEXT", "WEIGHT", "1.0", "NOSTEM",
		"text", "TEXT", "WEIGHT", "1.0", "NOSTEM",
		"embedding", "VECTOR", "HNSW", "6",
		"TYPE", "FLOAT64",
		"DIM", cfg.IndexDim,
		"DISTANCE_METRIC", cfg.DistanceMetric,
	).Err(); err != nil {
		return fmt.Errorf("Failed to create redis index: %s", err)
	}

	return nil
}

func (c *RedisClient) DropSchema(ctx context.Context, model Model) error {
	index := c.getIndexForModel(model)
	if _, err := c.getIndexInfo(ctx, index); err != nil {
		return fmt.Errorf("Failed to fetch index info: %s", err)
	}
	if err := c.rc.Do(ctx, "FT.DROPINDEX", index).Err(); err != nil {
		return fmt.Errorf("Failed to drop index: %s", err)
	}

	return nil
}

func (c *RedisClient) InsertDocument(ctx context.Context, model Model, doc *Document) error {
	vectorBytes := float64ToBytesLE(doc.Embedding)
	docKey := c.getDocKeyForModel(model, doc.FileId)

	if err := c.rc.HSet(ctx, docKey, map[string]any{
		"file_id":   doc.FileId,
		"text":      doc.Text,
		"embedding": vectorBytes,
	}).Err(); err != nil {
		return fmt.Errorf("Failed to insert a document to redis: %s", err)
	}

	return nil
}

func (c *RedisClient) InsertAllDocuments(ctx context.Context, model Model, docs []*Document) error {
	for _, doc := range docs {
		if err := c.InsertDocument(ctx, model, doc); err != nil {
			return err
		}
	}
	return nil
}

func (c *RedisClient) DeleteDocument(ctx context.Context, model Model, docKey string) error {
	if _, err := c.rc.Del(ctx, docKey).Result(); err != nil {
		return fmt.Errorf("Failed to delete key: %s", err)
	}
	return nil
}

func (c *RedisClient) DeleteAllDocuments(ctx context.Context, model Model) error {
	keys, err := c.rc.Keys(ctx, fmt.Sprintf("%s:*", c.getPrefixForModel(model))).Result()
	if err != nil {
		return fmt.Errorf("Failed to list keys: %s", err)
	}
	cnt := 0
	for _, docKey := range keys {
		if err := c.DeleteDocument(ctx, model, docKey); err != nil {
			return err
		}
		cnt++
	}
	log.Printf("Deleted %d keys", cnt)
	return nil
}

func (c *RedisClient) FindKNearest(ctx context.Context, model Model, doc *Document, k int) ([]*Document, error) {
	if doc.Embedding == nil {
		var vector []float64
		vector, err := c.emb.Embed(ctx, doc.Text)
		if err != nil {
			return nil, fmt.Errorf("Failed to embed text: %s", err)
		}
		doc.Embedding = vector
	}

	vectorBytes := float64ToBytesLE(doc.Embedding)
	index := c.getIndexForModel(model)
	res, err := c.rc.Do(ctx, "FT.SEARCH", index,
		fmt.Sprintf("*=>[KNN %d @embedding $vec_param AS title_score]", k),
		"PARAMS", "2", "vec_param", vectorBytes,
		"RETURN", "2", "file_id", "text",
		"DIALECT", "2",
	).Result()

	if err != nil {
		return nil, err
	}

	mm := res.(map[any]any)
	hits := mm["results"].([]any)
	docs := make([]*Document, 0, 1)
	for _, hit := range hits {
		dm := hit.(map[any]any)
		ea := dm["extra_attributes"].(map[any]any)
		fileId := ea["file_id"].(string)
		text := ea["text"].(string)
		doc := &Document{
			FileId: fileId,
			Text:   text,
		}
		log.Printf("Doc: %+v", doc)
		docs = append(docs, doc)
	}

	return docs, err
}

func (c *RedisClient) getDocKeyForModel(model Model, fileId string) string {
	return fmt.Sprintf("%s:{%s}", c.getPrefixForModel(model), fileId)
}

func (c *RedisClient) getIndexInfo(ctx context.Context, index string) (string, error) {
	return c.rc.Info(ctx, index).Result()
}

func (c *RedisClient) getIndexForModel(model Model) string {
	return fmt.Sprintf("idx:embeddings-%s", model.Name())
}

func (c *RedisClient) getPrefixForModel(model Model) string {
	return fmt.Sprintf("embeddings-%s:", model.Name())
}
