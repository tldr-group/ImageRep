import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext from "./interfaces";

import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';

const Topbar = () => {

    return (
        <Navbar bg={"dark"} variant="dark" expand="lg" style={{ boxShadow: "1px 1px  1px grey" }}>
            <Container>
                {/*path for these assets need to be relative to index.html in assets/*/}
                <Navbar.Brand><img src="favicon.png" width="40" height="40" className="d-inline-block align-top" /></Navbar.Brand>
                <Navbar.Brand style={{ marginLeft: "-25px", marginTop: "3px", fontSize: "1.75em" }}>epresentativity</Navbar.Brand>
                <Nav className="me-auto">a</Nav>
            </Container>
        </Navbar >
    );
}

export default Topbar;