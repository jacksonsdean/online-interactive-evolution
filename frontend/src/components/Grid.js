import * as React from "react";

import styles from "./Grid.module.css";

class Grid extends React.Component {

    constructor(props) {
        super(props);
    }

    render() {
        const isRow = this.props.row || !this.column;

        const classes =
        (!isRow ? styles.column : styles.row) +
        // Row styling
        (isRow && this.props.expanded ? ` ${styles.expanded}` : "") +
        (isRow && this.props.justify ? ` ${styles[this.props.justify]}` : "") +
        (isRow && this.props.alignItems ? ` ${styles["align-" + this.props.alignItems]}` : "") +
        // Column styling
        (!isRow && this.props.sm ? ` ${styles["sm-" + this.props.sm]}` : "") +
        (!isRow && this.props.md ? ` ${styles["md-" + this.props.md]}` : "") +
        (!isRow && this.props.lg ? ` ${styles["lg-" + this.props.lg]}` : "");

        return <div className={classes}>{this.props.children}</div>;
  };

}
  export default Grid;