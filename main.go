package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sandbox/embeddings/pkg/embeddings"

	"github.com/redis/go-redis/v9"
)

func main() {
	modelName := flag.String("model", "", "Ollama model to use")
	ollamaAddr := flag.String("ollama_addr", "http://127.0.0.1:11434", "Ollama address")
	indexDim := flag.Int("index_dim", 1024, "Index dimension")
	indexDist := flag.String("index_dist", "COSINE", "Index distance function")
	readFrom := flag.String("read_from", "", "Read embeddings from file")
	backend := flag.String("backend", "redis", "Backend to use")

	flag.Parse()

	command := flag.Arg(0)

	if len(*modelName) == 0 {
		log.Fatalf("Model name is required")
	}

	model, err := embeddings.ModelFromString(*modelName)
	if err != nil {
		log.Fatalf("Failed to parse model: %s", err)
	}

	log.Printf("Using ollama model: %s", model)

	httpc := &http.Client{}
	oe, err := embeddings.NewOllamaEmbedder(*ollamaAddr, httpc, model)
	if err != nil {
		log.Fatalf("Failed to create Ollama client: %s", err)
	}

	var ec embeddings.Client
	switch *backend {
	case "redis":
		ec, err = embeddings.NewRedisClient("localhost:6379", oe)
	case "chroma":
		ec, err = embeddings.NewChromaClient("http://localhost:35000", oe)
	}

	// Handle Ctrl-C
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		<-c
		log.Println("Exiting...")
		os.Exit(1)
	}()

	ctx := context.Background()
	switch command {
	case "repl":
		runRepl(ctx, ec, model)
	case "create_index":
		runCreateIndex(ctx, ec, model, &embeddings.SchemaConfig{
			IndexDim:       *indexDim,
			DistanceMetric: *indexDist,
		})
	case "drop_index":
		runDropIndex(ctx, ec, model)
	case "inject":
		runInject(ctx, ec, model, *readFrom)
	case "drop_keys":
		runDropKeys(ctx, ec, model)
	default:
		runHelp()
	}
}

func runDropKeys(ctx context.Context, ec embeddings.Client, model embeddings.Model) {
	if err := ec.DeleteAllDocuments(ctx, model); err != nil {
		log.Fatalf("Failed to delete all keys: %s", err)
	}
}

func runInject(ctx context.Context, ec embeddings.Client, model embeddings.Model, readFrom string) {
	if len(readFrom) == 0 {
		log.Fatalf("read_from is required")
	}
	f, err := os.Open(readFrom)
	if err != nil {
		log.Fatalf("Failed to open %s: %s", readFrom, err)
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	cnt := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				log.Printf("Done reading file")
				break
			} else {
				log.Fatalf("Failed to read from file: %s", err)
			}
		}
		if len(line) == 0 {
			continue
		}
		var doc embeddings.Document
		if err := json.Unmarshal([]byte(line), &doc); err != nil {
			log.Fatalf("Failed to unmarshal line: %s", err)
		}
		if len(doc.Embedding) == 0 {
			log.Printf("Skipping embedding without vector: %s", doc.FileId)
			continue
		}
		if allZero(doc.Embedding) {
			log.Printf("Skipping zero vector: %s", doc.FileId)
			continue
		}

		if err := ec.InsertDocument(ctx, model, &doc); err != nil {
			log.Printf("Failed to inject embedding: %s", err)
			log.Printf("Do you want to continue? [y/n]")
			var answer string
			fmt.Scanln(&answer)
			if answer != "y" {
				log.Printf("Exiting...")
				os.Exit(1)
			}
		} else {
			cnt++
		}
	}

	log.Printf("Injected %d records", cnt)
}

func injectEmbedding(ctx context.Context, ec embeddings.Client, model embeddings.Model, doc *embeddings.Document) error {
	if err := ec.InsertDocument(ctx, model, doc); err != nil {
		return err
	}

	return nil
}

func allZero(v []float64) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func getIndexForModel(model string) string {
	return fmt.Sprintf("idx:embeddings-%s", model)
}

func getPrefixForModel(model string) string {
	return fmt.Sprintf("embeddings-%s:", model)
}

func getIndexInfo(ctx context.Context, rc *redis.Client, index string) (string, error) {
	return rc.Info(ctx, index).Result()
}

func runDropIndex(ctx context.Context, ec embeddings.Client, model embeddings.Model) {
	if err := ec.DropSchema(ctx, model); err != nil {
		log.Fatalf("Failed to drop index: %s", err)
	}
}

func runCreateIndex(ctx context.Context, ec embeddings.Client, model embeddings.Model, cfg *embeddings.SchemaConfig) {
	if err := ec.CreateSchema(ctx, model, cfg); err != nil {
		log.Fatalf("Failed to create index: %s", err)
	}
}

func runHelp() {
	log.Println("Usage: TODO")
}

func runRepl(ctx context.Context, ec embeddings.Client, model embeddings.Model) {
	// Start REPL loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(">: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Failed to read input: %s", err)
			continue
		}
		fmt.Printf("Computing embedding vector...")
		doc := &embeddings.Document{
			Text: input,
		}
		knn, err := ec.FindKNearest(ctx, model, doc, 3)
		if err != nil {
			log.Printf("Failed to execute search query: %s", err)
		}
		for _, d := range knn {
			fmt.Printf("* %s\n", d.Text)
		}
	}
}
