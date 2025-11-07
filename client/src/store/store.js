import { configureStore, combineReducers } from '@reduxjs/toolkit'; // 1. Import combineReducers
import authReducer from '../features/auth/authSlice';
import {
  persistStore,
  persistReducer,
  FLUSH,
  REHYDRATE,
  PAUSE,
  PERSIST,
  PURGE,
  REGISTER,
} from 'redux-persist';
import storage from 'redux-persist/lib/storage';

// Configuration for redux-persist
const persistConfig = {
  key: 'root',
  storage,
  whitelist: ['auth'], // This will now correctly target the 'auth' slice
};

// 2. Create the root reducer by combining your slices first
const rootReducer = combineReducers({
  auth: authReducer,
  // You can add other non-persisted reducers here:
  // posts: postsReducer,
});

// 3. Create the persisted reducer
const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  // 4. Use the single persistedReducer
  reducer: persistedReducer,
  // Configure middleware to ignore Redux Persist actions
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
      },
    }),
});

// Create the persistor
export const persistor = persistStore(store);