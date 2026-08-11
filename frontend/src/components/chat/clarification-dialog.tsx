"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Send,
  X,
  Lightbulb,
} from "lucide-react";

interface ClarificationDialogProps {
  question: string;
  options?: string[];
  rationale?: string;
  onRespond: (response: string) => void;
  onSkip: () => void;
  attempt?: number;
  maxAttempts?: number;
}

export default function ClarificationDialog({
  question,
  options = [],
  rationale = "",
  onRespond,
  onSkip,
  attempt = 1,
  maxAttempts = 2,
}: ClarificationDialogProps) {
  const [inputValue, setInputValue] = useState("");
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const response = selectedOption || inputValue.trim();
      if (response) {
        onRespond(response);
      }
    },
    [selectedOption, inputValue, onRespond]
  );

  const handleOptionSelect = useCallback(
    (option: string) => {
      setSelectedOption(option);
      setInputValue(option);
    },
    []
  );

  const handleSkip = useCallback(() => {
    onSkip();
  }, [onSkip]);

  return (
    <div className="my-4 rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/30 p-4">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center">
            <Lightbulb className="w-4 h-4 text-blue-600 dark:text-blue-400" />
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-blue-900 dark:text-blue-100">
              Clarification Needed
            </span>
            {attempt > 1 && (
              <Badge variant="secondary" className="text-xs">
                Attempt {attempt}/{maxAttempts}
              </Badge>
            )}
          </div>

          <p className="text-sm text-blue-800 dark:text-blue-200 mb-3">
            {question}
          </p>

          {rationale && (
            <p className="text-xs text-blue-600 dark:text-blue-400 mb-3 italic">
              {rationale}
            </p>
          )}

          {options.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {options.map((option, idx) => (
                <Button
                  key={idx}
                  variant={selectedOption === option ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleOptionSelect(option)}
                  className={`text-xs ${
                    selectedOption === option
                      ? "bg-blue-600 hover:bg-blue-700 text-white"
                      : "border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/30"
                  }`}
                >
                  {option}
                </Button>
              ))}
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              type="text"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                setSelectedOption(null);
              }}
              placeholder="Type your answer..."
              className="flex-1 text-sm bg-white dark:bg-gray-900 border-blue-200 dark:border-blue-800"
              autoFocus
            />
            <Button
              type="submit"
              size="sm"
              disabled={!inputValue.trim() && !selectedOption}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Send className="w-3 h-3 mr-1" />
              Send
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleSkip}
              className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            >
              <X className="w-3 h-3" />
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}
