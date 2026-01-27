FROM node:18-bullseye
WORKDIR /frontend
COPY . .
ENV REACT_APP_NODE_ENV="local"
RUN npm ci
CMD ["npm", "start"]
EXPOSE 3000
EXPOSE 3001
