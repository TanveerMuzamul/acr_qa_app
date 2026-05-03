// Handles login form submission and authentication requests
// Front-end controller for the MRI ACR QA application.
// Comments mark the main UI update and API communication steps.

let mode = 'login';

const form = document.getElementById('authForm');
const loginTab = document.getElementById('loginTab');
const registerTab = document.getElementById('registerTab');
const title = document.getElementById('authTitle');
const subtitle = document.getElementById('authSubtitle');
const submitBtn = document.getElementById('submitBtn');
const errorBox = document.getElementById('loginError');

function setMode(nextMode) {
  mode = nextMode;
  const isLogin = mode === 'login';
  loginTab.classList.toggle('active', isLogin);
  registerTab.classList.toggle('active', !isLogin);
  title.textContent = isLogin ? 'Welcome back' : 'Create account';
  subtitle.textContent = isLogin
    ? 'Sign in to open the MRI QA dashboard.'
    : 'Register a new local account for this app.';
  submitBtn.textContent = isLogin ? 'Login' : 'Register';
  errorBox.textContent = '';
}

loginTab.onclick = () => setMode('login');
registerTab.onclick = () => setMode('register');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorBox.textContent = '';
  submitBtn.disabled = true;

  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;
  const endpoint = mode === 'login' ? '/api/login' : '/api/register';

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const result = await response.json();
    if (!response.ok) {
      errorBox.textContent = result.error || 'Access failed';
      return;
    }
    window.location.href = '/';
  } finally {
    submitBtn.disabled = false;
  }
});
