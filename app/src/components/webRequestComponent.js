import React from 'react';

class WebRequestComponent extends React.Component {

  constructor(props) {
    super(props);
    this.state = {value: "Please wait..."};
  }

  componentDidMount() {
    const apiUrl = 'https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/test/interactive-evolutionary-computation';
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => this.setState({ value: data }));
  }

  render() {
    return <h1>{this.state.value}</h1>;
  }
}
export default WebRequestComponent;