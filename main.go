package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"

	// "github.com/RediSearch/redisearch-go/redisearch"
	ollama "github.com/ollama/ollama/api"
	"github.com/redis/go-redis/v9"
)

func main() {
	model := flag.String("model", "", "Ollama model to use")
	ollamaAddr := flag.String("ollama_addr", "http://127.0.0.1:11434", "Ollama address")
	indexDim := flag.Int("index_dim", 1024, "Index dimension")
	indexDist := flag.String("index_dist", "COSINE", "Index distance function")

	flag.Parse()

	command := flag.Arg(0)

	if len(*model) == 0 {
		log.Fatalf("Model is required")
	}

	log.Printf("Using ollama model: %s", *model)

	ctx := context.Background()
	httpc := &http.Client{}
	url, err := url.Parse(*ollamaAddr)
	if err != nil {
		log.Fatalf("Failed to parse ollama address: %s", err)
	}
	ollamaClient := ollama.NewClient(url, httpc)
	if v, err := ollamaClient.Version(ctx); err != nil {
		log.Fatalf("Failed to get Ollama version: %s", err)
	} else {
		log.Printf("Ollama version: %s", v)
	}

	rdb := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})
	//rsearch := redisearch.NewClient("localhost:6379", index)
	pong := rdb.Ping(ctx)
	if pong.Err() != nil {
		log.Fatalf("Failed to ping Redis: %s", pong.Err())
	}
	log.Printf("Redis ping: %s", pong.Val())

	// Handle Ctrl-C
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		<-c
		log.Println("Exiting...")
		os.Exit(1)
	}()

	switch command {
	case "repl":
		runRepl(ctx, rdb, ollamaClient, *model)
	case "create_index":
		runCreateIndex(ctx, rdb, *model, *indexDim, *indexDist)
	case "drop_index":
		runDropIndex(ctx, rdb, *model)
	default:
		runHelp()
	}
}

func getIndexForModel(model string) string {
	return fmt.Sprintf("idx:embeddings-%s", model)
}

func getIndexInfo(ctx context.Context, rc *redis.Client, index string) (string, error) {
	return rc.Info(ctx, index).Result()
}

func runDropIndex(ctx context.Context, rc *redis.Client, model string) {
	index := getIndexForModel(model)
	if _, err := getIndexInfo(ctx, rc, index); err != nil {
		log.Fatalf("Failed to fetch index info: %s", err)
	}
	if err := rc.Do(ctx, "FT.DROPINDEX", index).Err(); err != nil {
		log.Fatalf("Failed to drop index: %s", err)
	}
	log.Printf("Index %s successfully dropped", index)
}

func runCreateIndex(ctx context.Context, rc *redis.Client, model string, dim int, dist string) {
	index := getIndexForModel(model)
	log.Printf("Creating index %s", index)
	indexInfo, err := getIndexInfo(ctx, rc, index)
	if err != nil {
		log.Fatalf("Failed to fetch index info: %s", err)
	}
	if len(indexInfo) != 0 {
		log.Fatalf("Index %s already exists", index)
	}

	_, err = rc.Do(ctx, "FT.CREATE", index,
		"SCHEMA",
		"file_id", "TEXT", "WEIGHT", "1.0", "NOSTEM",
		"text", "TEXT", "WEIGHT", "1.0", "NOSTEM",
		"embedding", "VECTOR", "HNSW", "6", "TYPE", "FLOAT64", "DIM", dim, "DISTANCE_METRIC", dist,
	).Result()

	if err != nil {
		log.Fatalf("Failed to create index: %s", err)
	}

	log.Printf("Index created: %s", index)
}

func runHelp() {
	log.Println("Usage: TODO")
}

func runRepl(ctx context.Context, rc *redis.Client, oc *ollama.Client, model string) {
	// Start REPL loop
	index := getIndexForModel(model)
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(">: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Failed to read input: %s", err)
			continue
		}
		fmt.Printf("Compute embedding vector...")
		embed, err := computeEmbedding(ctx, oc, model, input)
		if err != nil {
			log.Printf("Failed to compute embedding: %s", err)
			continue
		}
		fmt.Println("done.")

		res, err := findSimilar(ctx, rc, index, embed, 3)
		if err != nil {
			log.Printf("Failed to execute search query: %s", err)
			continue
		}
		log.Printf("Result: %+v", res)
	}
}

// Encode a slice of float64 values as a byte slice (little-endian)
func float64ToBytesLE(floats []float64) []byte {
	var bytes []byte
	for _, f := range floats {
		bits := math.Float64bits(f)
		bytes = append(bytes,
			byte(bits), // Least significant byte first
			byte(bits>>8),
			byte(bits>>16),
			byte(bits>>24),
			byte(bits>>32),
			byte(bits>>40),
			byte(bits>>48),
			byte(bits>>56), // Most significant byte last
		)
	}
	return bytes
}

func findSimilar(ctx context.Context, client *redis.Client, index string, vector []float64, numRes int) (any, error) {
	vectorBytes := float64ToBytesLE(vector)
	res, err := client.Do(ctx, "FT.SEARCH", index,
		"*",                                     // Search all documents
		"PARAMS", "2", "vec_param", vectorBytes, // Vector query params
		"DIALECT", "2", // Optional, using a specific dialect version if needed
		"RETURN", "2", "$.file_id", "$.text", // Specify fields to return (e.g., title)
		"LIMIT", "0", strconv.Itoa(numRes), // Limit the result set
	).Result()
	return res, err
}

func computeEmbedding(ctx context.Context, oc *ollama.Client, model string, prompt string) ([]float64, error) {
	e, err := oc.Embeddings(ctx, &ollama.EmbeddingRequest{
		Model:  model,
		Prompt: prompt,
	})
	if err != nil {
		return nil, err
	}
	return e.Embedding, nil
}
