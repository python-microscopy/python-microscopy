<template>
  <div class="login-container">
    <form class="login-form" :class="{ error: hasError }" @submit.prevent="handleLogin">
      <h3>Sign in to PYME Acquire</h3>
      
      <div v-if="errorMessage" class="error-message">
        {{ errorMessage }}
      </div>
      
      <div class="form-group">
        <label for="email">Email:</label>
        <input
          type="text"
          id="email"
          v-model="email"
          placeholder="Email address"
          required
          :disabled="isLoading"
        />
      </div>
      
      <div class="form-group">
        <label for="password">Password:</label>
        <input
          type="password"
          id="password"
          v-model="password"
          placeholder="Password"
          required
          :disabled="isLoading"
        />
      </div>
      
      <button type="submit" :disabled="isLoading">
        {{ isLoading ? 'Signing in...' : 'Submit' }}
      </button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const email = ref('')
const password = ref('')
const isLoading = ref(false)
const errorMessage = ref('')
const hasError = ref(false)

async function handleLogin() {
  isLoading.value = true
  errorMessage.value = ''
  hasError.value = false
  
  const result = await authStore.login(email.value, password.value)
  
  if (result.success) {
    // Redirect to the original destination or home
    const redirect = route.query.redirect || '/'
    router.push(redirect)
  } else {
    errorMessage.value = result.error || 'Email or password not valid'
    hasError.value = true
    isLoading.value = false
  }
}
</script>

<style scoped lang="scss">
.login-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.login-form {
  width: 100%;
  max-width: 400px;
  padding: 2rem;
  background-color: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: border-color 0.3s, background-color 0.3s;
}

.login-form.error {
  border: 2px solid #dc3545;
  background-color: #fff5f5;
}

.login-form h3 {
  margin-bottom: 1.5rem;
  text-align: center;
  color: #333;
}

.error-message {
  padding: 0.75rem;
  margin-bottom: 1rem;
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
}

/* Override global form-group styles for light theme */
.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #333;
}

.form-group input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  background-color: white;
  color: #333;
}

.form-group input:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

button {
  width: 100%;
  padding: 0.75rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:hover:not(:disabled) {
  background-color: #0056b3;
}

button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}
</style>
