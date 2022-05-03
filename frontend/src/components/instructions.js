import React, { useState } from "react";
import useCollapse from 'react-collapsed'
import styles from "./Instructions.module.css"

function Instructions() {
    const [isExpanded, setExpanded] = useState(false)
    const { getCollapseProps, getToggleProps } = useCollapse({ isExpanded })

    return (
      <div className={styles.instructions}>
        <button className={styles.instructionsButton }
          {...getToggleProps({
            onClick: () => setExpanded((prevExpanded) => !prevExpanded),
          })}
        >
          {isExpanded ? 'Instructions \u25B2' : 'Instructions \u25Bc'}
        </button>
        <section {...getCollapseProps()}>Instructions here 🙈</section>
      </div>
    )
  }
export default Instructions;
