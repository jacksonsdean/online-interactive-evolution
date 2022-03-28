import React from 'react';

class WebRequestComponent extends React.Component {
  componentDidMount() {
    const apiUrl = 'https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/test/helloworld?name=John&city=Seattle';
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => console.log('This is your data', data));
  }
  render() {
    return <h1></h1>;
  }
}
export default WebRequestComponent;