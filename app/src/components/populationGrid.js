import React from 'react';


class IndividualButton extends React.Component {
    render(){
        return(
            <button className="individual-button" onClick={this.props.onClick}>
                <img src={this.props.image} alt={this.props.alt}/>
            </button>
        )
    }
}

class PopulationGrid extends React.Component {
    componentDidMount() {

    }
    render() {
        return (<div>

        </div>
        );
    }
}
export default PopulationGrid;