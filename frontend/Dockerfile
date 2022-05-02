FROM node:12
WORKDIR /frontend
COPY . .
ENV REACT_APP_NODE_ENV="local"
RUN npm ci
CMD ["npm", "start"]
EXPOSE 3000
EXPOSE 3001
