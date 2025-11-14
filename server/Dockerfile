# Use a secure, official Node.js base image (e.g., Node 20)
FROM node:20-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install production dependencies
# 'npm ci' is faster and more reliable for CI/CD
RUN npm ci --only=production

# Copy the rest of your application code
COPY . .

# Expose the port your app runs on
# ----------------------------------------------------
# !!! IMPORTANT !!!
# Change 8080 to whatever port your app listens on
# (e.g., the 'port' variable in your server.js)
# ----------------------------------------------------
EXPOSE 8080

# The command to start your application
# (Change 'server.js' if your main file is named 'index.js' or similar)
CMD [ "node", "index.js" ]