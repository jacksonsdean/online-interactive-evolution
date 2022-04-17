import React from 'react';


function to_ids(a){
    var b = a.split(',').map(function(item) {
        return parseInt(item, 10);
    });
    return b;
}

class IndividualButton extends React.Component {
    render() {
        return (
            <button className="individual-button" onClick={this.props.onClick}>
                {/* <img src={this.props.image} alt={this.props.alt} /> */}
                {this.props.name}
            </button>
        )
    }
}

class PopulationGrid extends React.Component {

    constructor(props) {
        super(props);
        this.state = { population_ids: ["Please wait.."] };
    }

    componentDidMount() {
        const apiUrl = 'https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/test/interactive-evolutionary-computation?ids=1,2,3,4,5,10';
        fetch(apiUrl)
            .then((response) => response.json())
            .then((data) => this.setState({ population_ids: to_ids(data) }));
    }

    render() {
        // list of all the population ids
        return (<div>
            <ul>
            {this.state.population_ids.map(i => <li key={i}> <IndividualButton name={i}></IndividualButton> </li>)}
            </ul>
        </div>
        );
    }
}
export default PopulationGrid;