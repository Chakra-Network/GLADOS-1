package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"syscall"
)

type Action struct {
	SessionID           string `json:"session_id"`
	ScreenshotURL       string `json:"screenshot_url"`
	RelativeTimestampMs int64  `json:"relative_timestamp_ms"`
}

const REQUIRED_DISK_SPACE_BUFFER_BYTES = 50 * 1024 * 1024 * 1024 // 50GB

func main() {
	var screenshotPath string
	var actionsPath string
	var maxConcurrent int

	flag.StringVar(&screenshotPath, "screenshot_path", "", "Path to save screenshots")
	flag.StringVar(&actionsPath, "actions_path", "", "Path to JSON file containing actions")
	flag.IntVar(&maxConcurrent, "max_concurrent", 10, "Maximum concurrent downloads")
	flag.Parse()

	if screenshotPath == "" || actionsPath == "" {
		fmt.Println("Usage: go run main.go -screenshot_path <path> -actions_path <path> [-max_concurrent <num>]")
		os.Exit(1)
	}

	if err := os.MkdirAll(screenshotPath, 0755); err != nil {
		fmt.Printf("Error creating screenshot directory: %v\n", err)
		os.Exit(1)
	}

	actions, err := readActions(actionsPath)
	if err != nil {
		fmt.Printf("Error reading actions file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Found %d actions to process\n", len(actions))

	if err := downloadScreenshots(actions, screenshotPath, maxConcurrent); err != nil {
		fmt.Printf("Error downloading screenshots: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("All screenshots downloaded successfully")
}

func readActions(actionsPath string) ([]Action, error) {
	file, err := os.Open(actionsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open actions file: %w", err)
	}
	defer file.Close()

	var actions []Action
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&actions); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return actions, nil
}

func downloadScreenshots(actions []Action, screenshotPath string, maxConcurrent int) error {
	semaphore := make(chan struct{}, maxConcurrent)

	var wg sync.WaitGroup
	var mu sync.Mutex
	var errors []error
	var totalBytes int64

	for i, action := range actions {
		wg.Add(1)
		go func(idx int, act Action) {
			defer wg.Done()

			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			numBytes, err := downloadScreenshot(act, screenshotPath)
			if err != nil {
				mu.Lock()
				errors = append(errors, fmt.Errorf("failed to download screenshot for action %s: %w", act.SessionID, err))
				mu.Unlock()
				return
			}
			mu.Lock()
			totalBytes += numBytes
			mu.Unlock()
			if numBytes > 0 {
				fmt.Printf("Downloaded screenshot %d/%d, total GB: %d\n", idx+1, len(actions), totalBytes/(1024*1024*1024))
			}
		}(i, action)
	}

	wg.Wait()

	if len(errors) > 0 {
		fmt.Printf("Encountered %d errors during download:\n", len(errors))
		for _, err := range errors {
			fmt.Printf("  %v\n", err)
		}
		return fmt.Errorf("failed to download %d screenshots", len(errors))
	}

	return nil
}

func downloadScreenshot(action Action, screenshotPath string) (int64, error) {
	filename := fmt.Sprintf("%s_%d.png", action.SessionID, action.RelativeTimestampMs)
	filePath := filepath.Join(screenshotPath, filename)

	if _, err := os.Stat(filePath); err == nil {
		return 0, nil
	}

	client := &http.Client{}

	resp, err := client.Get(action.ScreenshotURL)
	if err != nil {
		return 0, fmt.Errorf("failed to fetch screenshot: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP request failed with status: %s", resp.Status)
	}

	checkDiskSpace(screenshotPath)

	file, err := os.Create(filePath)
	if err != nil {
		return 0, fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	numBytes, err := io.Copy(file, resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to write file: %w", err)
	}

	return numBytes, nil
}

func checkDiskSpace(path string) error {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		panic(fmt.Errorf("failed to get disk space for %s: %w", path, err))
	}

	// Available space = available blocks * block size
	availableBytes := stat.Bavail * uint64(stat.Bsize)
	if availableBytes < REQUIRED_DISK_SPACE_BUFFER_BYTES {
		panic(fmt.Sprintf("insufficient disk space: need ~%d, have %d available", REQUIRED_DISK_SPACE_BUFFER_BYTES, availableBytes))
	}

	return nil
}
