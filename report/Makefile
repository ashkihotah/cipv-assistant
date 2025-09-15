# Makefile for compiling the LaTeX document

# Variables
LATEXMK = latexmk
MAIN_FILE = presentation
OUT_DIR = presentation_out

# latexmk flags
# -lualatex: Use LuaLaTeX as the compiler
# -shell-escape: Needed for packages like 'minted'
# -output-directory: Place generated files in the 'out' folder
# -pdf: Generate a PDF file
LATEXMK_FLAGS = -lualatex -shell-escape -output-directory=$(OUT_DIR) -pdf

# Default target, executed when you just run 'make'
all: compile

# The main compilation rule using latexmk
# latexmk automatically handles dependencies and recompiles only when needed.
compile:
	@echo "--- Starting LaTeX compilation with latexmk ---"
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN_FILE).tex
	@echo "--- Compilation finished successfully! ---"
	@echo "PDF is at: $(OUT_DIR)/$(MAIN_FILE).pdf"

# Target to clean up all generated files
clean:
	@echo "--- Cleaning up generated files ---"
	$(LATEXMK) -C
	@if exist $(OUT_DIR) ( rmdir /s /q $(OUT_DIR) )
	@echo "--- Cleanup complete ---"

# Phony targets don't represent actual files
.PHONY: all compile clean
