"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import { APP_NAME } from "@/lib/app-config";
import { validatePasswordStrength } from "@/lib/auth";

export default function RegisterPage() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [validationErrors, setValidationErrors] = useState({
    email: "",
    password: "",
    confirmPassword: "",
  });

  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setValidationErrors((prev) => ({ ...prev, email: "Please enter a valid email address" }));
      return false;
    }
    setValidationErrors((prev) => ({ ...prev, email: "" }));
    return true;
  };

  const validatePassword = (password: string) => {
    const error = validatePasswordStrength(password);
    if (error) {
      setValidationErrors((prev) => ({ ...prev, password: error }));
      return false;
    }
    setValidationErrors((prev) => ({ ...prev, password: "" }));
    return true;
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setValidationErrors({ email: "", password: "", confirmPassword: "" });

    const formData = new FormData(e.currentTarget);
    const username = formData.get("username") as string;
    const email = formData.get("email") as string;
    const password = formData.get("password") as string;
    const confirmPassword = formData.get("confirmPassword") as string;

    const isEmailValid = validateEmail(email);
    const isPasswordValid = validatePassword(password);

    if (password !== confirmPassword) {
      setValidationErrors((prev) => ({ ...prev, confirmPassword: "Passwords do not match" }));
      return;
    }
    if (!isEmailValid || !isPasswordValid) return;

    try {
      await api.post("/api/auth/register", { username, email, password });
      router.push("/");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Registration failed");
      }
    }
  };

  const inputBase = "w-full px-3 py-2 rounded-md border bg-background text-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground";

  return (
    <main className="min-h-screen bg-background flex items-center justify-center px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-md">
        <div className="bg-card text-card-foreground rounded-lg border shadow-md p-8 space-y-6">
          <div className="text-center">
            <h1 className="text-3xl font-bold">Welcome To {APP_NAME}</h1>
            <p className="mt-2 text-sm text-muted-foreground">Create your account to get started</p>
          </div>

          <form className="space-y-6" onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <label htmlFor="username" className="block text-sm font-medium mb-1">Username</label>
                <input id="username" name="username" type="text" required
                  className={inputBase + " border-input"}
                  placeholder="Enter your username" />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium mb-1">Email</label>
                <input id="email" name="email" type="email" required
                  className={inputBase + (validationErrors.email ? " border-destructive" : " border-input")}
                  placeholder="Enter your email"
                  onChange={(e) => validateEmail(e.target.value)} />
                {validationErrors.email && (
                  <p className="mt-1 text-sm text-destructive">{validationErrors.email}</p>
                )}
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium mb-1">Password</label>
                <input id="password" name="password" type="password" required
                  className={inputBase + (validationErrors.password ? " border-destructive" : " border-input")}
                  placeholder="Create a password"
                  onChange={(e) => validatePassword(e.target.value)} />
                {validationErrors.password && (
                  <p className="mt-1 text-sm text-destructive">{validationErrors.password}</p>
                )}
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium mb-1">Confirm Password</label>
                <input id="confirmPassword" name="confirmPassword" type="password" required
                  className={inputBase + (validationErrors.confirmPassword ? " border-destructive" : " border-input")}
                  placeholder="Confirm your password" />
                {validationErrors.confirmPassword && (
                  <p className="mt-1 text-sm text-destructive">{validationErrors.confirmPassword}</p>
                )}
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-md bg-destructive/10 text-destructive text-sm">{error}</div>
            )}

            <button type="submit"
              className="w-full flex justify-center py-2 px-4 rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-ring transition-colors">
              Create Account
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
