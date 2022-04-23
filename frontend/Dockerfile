FROM node:12
WORKDIR /frontend
COPY . .
RUN npm ci
CMD ["npm", "start"]
EXPOSE 3000