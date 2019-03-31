package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
)

func execute(fpath string, password string, num int) {
	cmd := exec.Command("python3", fpath+"solve_task.py", password, strconv.Itoa(num))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}

func main() {
	fpath := os.Args[1]
	password := os.Args[2]
	for i := 113; i < 117; i++ {
		execute(fpath, password, i)
	}
	fmt.Scanln()
}
