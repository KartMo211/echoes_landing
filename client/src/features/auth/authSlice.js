import { createSlice } from '@reduxjs/toolkit';

// Initial state will be empty
const initialState = {
  username: null,
  token: null, // This is where the JWT will be stored
  isAuthenticated: false,
};

export const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    // Action to set credentials on successful login
    setCredentials: (state, action) => {
      const { username, token } = action.payload;
      state.username = username;
      state.token = token;
      state.isAuthenticated = true;
    },
    // Action to clear credentials on logout
    logOut: (state) => {
      state.username = null;
      state.token = null;
      state.isAuthenticated = true;
    },
  },
});



export const { setCredentials, logOut } = authSlice.actions;

export default authSlice.reducer;
