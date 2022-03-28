
function testRestAPI() {
    fetch('https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/test/helloworld?name=John&city=Seattle')
    .then(response => response.json())
    .then(data => console.log(data));

}

export { testRestAPI };