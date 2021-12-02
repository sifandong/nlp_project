import {
  Grid,
  Container,
  Stack,
  TextField,
  Button,
  Typography,
} from "@mui/material";

function Home() {
  return (
    <Container maxWidth="md" style={{ height: "100vh" }}>
      <Grid direction="column" container>
        <Grid
          item
          container
          justifyContent="center"
          alignItems="center"
          style={{ height: "10%", marginTop: "24px" }}
        >
          <Typography variant="h4" gutterBottom component="div">
            Group 8 Presentation
          </Typography>
        </Grid>
        <Grid
          item
          container
          justifyContent="center"
          alignItems="center"
          style={{ height: "10%", marginTop: "24px" }}
        >
          <TextField
            fullWidth
            id="outlined-basic"
            label="Link"
            variant="outlined"
          />
        </Grid>
        <Grid
          item
          container
          justifyContent="center"
          alignItems="center"
          style={{ height: "40%", marginTop: "24px" }}
        >
          <TextField
            fullWidth
            id="outlined-multiline-static"
            label="Reviews Text"
            multiline
            rows={4}
            // defaultValue="Default Value"
          />
        </Grid>
        <Grid
          item
          container
          justifyContent="center"
          alignItems="center"
          style={{ height: "10%", marginTop: "24px" }}
        >
          <Button variant="contained">Submit</Button>
        </Grid>
      </Grid>
    </Container>
  );
}

export default Home;
